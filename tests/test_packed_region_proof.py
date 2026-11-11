"""Structured slot congruence checks guards, binders, captures and termination."""

import pytest

from t2m.diagnostics import ConversionError
from t2m.elementwise_map import expand_elementwise_maps
from t2m.shared_tensor import _source_ids, parse_shared


def module(body, returns="%r0,%r1"):
    return parse_shared('''module {
      func.func @probe(%x:tensor<8xi64>,%limit:index,%flag:i1)->tensor<8xi64> {
        %zero=arith.constant 0:index
        %one=arith.constant 1:index
        %v="tt.map_elementwise"(%x) <{pack=2:i32}> ({
        ^bb0(%a0:i64,%a1:i64):
    ''' + body + '''
          "tt.map_elementwise.return"(''' + returns + '''):(i64,i64)->()
        }):(tensor<8xi64>)->tensor<8xi64>
        func.return %v:tensor<8xi64>
      }
    }''')


def test_nested_captures_keep_their_original_producers_and_all_slot_locations():
    value = module('''
      %b0=arith.addi %a0,%a0:i64 loc("ttir.op.producer0")
      %b1=arith.addi %a1,%a1:i64 loc("ttir.op.producer1")
      %r0=scf.if %flag -> (i64) {
        %q0=arith.subi %b0,%a0:i64 loc("ttir.op.nested0")
        scf.yield %q0:i64
      } else {scf.yield %a0:i64}
      %r1=scf.if %flag -> (i64) {
        %q1=arith.subi %b1,%a1:i64 loc("ttir.op.nested1")
        scf.yield %q1:i64
      } else {scf.yield %a1:i64}
    ''')
    report, = expand_elementwise_maps(value, _source_ids)
    value.verify()
    assert report["packing_proof"]["method"] == "typed-structured-ssa-congruence-v1"
    producers = [op for op in value.walk() if op.name == "arith.addi"]
    nested = [op for op in value.walk() if op.name == "arith.subi"]
    assert len(producers) == len(nested) == 1
    assert set(_source_ids(producers[0].location)) == {"ttir.op.producer0", "ttir.op.producer1"}
    assert set(_source_ids(nested[0].location)) == {"ttir.op.nested0", "ttir.op.nested1"}
    assert nested[0].operands[0] is producers[0].results[0]


@pytest.mark.parametrize("change", [None, "guard", "capture", "branch-order"])
def test_nested_slot_dependencies_cannot_hide_behind_equal_scalar_terms(change):
    guard0 = "%a1" if change == "guard" else "%a0"
    guard1 = "%a0" if change == "guard" else "%a1"
    operand0 = "%a1" if change == "capture" else "%a0"
    operand1 = "%a0" if change == "capture" else "%a1"
    branch1 = "%a1" if change == "branch-order" else "%b1"
    other1 = "%b1" if change == "branch-order" else "%a1"
    value = module(f'''
      %bound=arith.index_cast %limit:index to i64
      %p0=arith.cmpi ne,{guard0},%bound:i64
      %p1=arith.cmpi ne,{guard1},%bound:i64
      %b0=arith.addi %a0,{operand0}:i64
      %b1=arith.addi %a1,{operand1}:i64
      %r0=scf.if %p0 -> (i64) {{scf.yield %b0:i64}}
      else {{scf.yield %a0:i64}}
      %r1=scf.if %p1 -> (i64) {{scf.yield {branch1}:i64}}
      else {{scf.yield {other1}:i64}}
    ''')
    value.verify()
    if change is None:
        expand_elementwise_maps(value, _source_ids)
        value.verify()
        assert sum(op.name == "scf.if" for op in value.walk()) == 1
    else:
        with pytest.raises(ConversionError, match="logical grouping.*coupled|logical grouping.*different"):
            expand_elementwise_maps(value, _source_ids)


@pytest.mark.parametrize("inner_capture", [False, True])
def test_lexical_loop_binders_are_alpha_equivalent_but_cannot_alias(inner_capture):
    second = "%j1" if inner_capture else "%i1"
    body = []
    for slot, index in ((0, "%i0"), (1, second)):
        body.append(f'''
          %r{slot}=scf.for %i{slot}=%zero to %limit step %one iter_args(%outer{slot}=%a{slot})->(i64) {{
            %inner{slot}=scf.for %j{slot}=%zero to %limit step %one iter_args(%carry{slot}=%outer{slot})->(i64) {{
              %word{slot}=arith.index_cast {index}:index to i64
              %next{slot}=arith.addi %carry{slot},%word{slot}:i64
              scf.yield %next{slot}:i64
            }}
            scf.yield %inner{slot}:i64
          }}
        ''')
    value = module("\n".join(body))
    if inner_capture:
        with pytest.raises(ConversionError, match="different scalar functions"):
            expand_elementwise_maps(value, _source_ids)
    else:
        expand_elementwise_maps(value, _source_ids)
        value.verify()
        assert sum(op.name == "scf.for" for op in value.walk()) == 2


def test_multiple_loop_results_retain_their_original_result_positions():
    body = []
    for slot in (0, 1):
        body.append(f'''
          %p{slot}:2=scf.for %i{slot}=%zero to %limit step %one
            iter_args(%left{slot}=%a{slot},%right{slot}=%a{slot})->(i64,i64) {{
              %next{slot}=arith.addi %left{slot},%right{slot}:i64
              scf.yield %next{slot},%right{slot}:i64,i64
          }}
        ''')
    value = module("\n".join(body), returns="%p0#0,%p1#1")
    with pytest.raises(ConversionError, match="different scalar functions"):
        expand_elementwise_maps(value, _source_ids)


def test_unused_while_is_not_erased_as_dead_arithmetic():
    value = module('''
      %unused=scf.while (%state=%a0):(i64)->i64 {
        scf.condition(%flag) %state:i64
      } do {^bb0(%state:i64):scf.yield %state:i64}
    ''', returns="%a0,%a1")
    with pytest.raises(ConversionError, match="execution/termination retained"):
        expand_elementwise_maps(value, _source_ids)


@pytest.mark.parametrize("flags", ["", " fastmath<reassoc>"])
def test_nested_arithmetic_attributes_remain_executable_semantics(flags):
    value = parse_shared('''module {
      func.func @attributes(%x:tensor<8xf32>,%flag:i1)->tensor<8xf32> {
        %v="tt.map_elementwise"(%x) <{pack=2:i32}> ({
        ^bb0(%a0:f32,%a1:f32):
          %r0=scf.if %flag -> (f32) {
            %p=arith.addf %a0,%a0:f32
            scf.yield %p:f32
          } else {scf.yield %a0:f32}
          %r1=scf.if %flag -> (f32) {
            %q=arith.addf %a1,%a1''' + flags + ''':f32
            scf.yield %q:f32
          } else {scf.yield %a1:f32}
          "tt.map_elementwise.return"(%r0,%r1):(f32,f32)->()
        }):(tensor<8xf32>)->tensor<8xf32>
        func.return %v:tensor<8xf32>
      }
    }''')
    value.verify()
    if flags:
        with pytest.raises(ConversionError, match="different scalar functions"):
            expand_elementwise_maps(value, _source_ids)
    else:
        expand_elementwise_maps(value, _source_ids)
        value.verify()
