"""Relative addresses need actual common SSA roots, exact units and no poison."""

import pytest
from test_pointer_addresses import _module
from xdsl.dialects import arith, builtin

from t2m.diagnostics import ConversionError
from t2m.ir import text
from t2m.pointer_addresses import normalize_pointer_addresses
from t2m.shared_tensor import _source_ids, opcode, parse_shared


def unchanged_refusal(module, message):
    before = text(module)
    values = [value for operation in module.walk() for value in (
        *operation.results, *(arg for region in operation.regions for block in region.blocks for arg in block.args))]
    uses = {value: set(value.uses) for value in values}
    with pytest.raises(ConversionError, match=message):
        normalize_pointer_addresses(module, _source_ids)
    assert text(module) == before
    assert {value: set(value.uses) for value in values} == uses


def test_cancelled_byte_values_can_escape_without_exposing_the_original_base():
    module = _module('''%first=arith.addi %bits,%offset:i64
      %second=arith.subi %bits,%offset:i64
      %delta=arith.subi %first,%second:i64 loc("ttir.op.difference")
      "tt.store"(%out,%delta):(!tt.ptr<i64>,i64)->()''')
    reports = normalize_pointer_addresses(module, _source_ids)
    module.verify()
    assert len(reports) == 1 and reports[0]["kind"] == "closed-pointer-integer-difference"
    assert reports[0]["units"] == "bytes" and "ttir.op.difference" in reports[0]["source_ids"]
    assert not any(opcode(op) in ("tt.ptr_to_int", "tt.int_to_ptr") for op in module.walk())
    assert next(op for op in module.walk() if opcode(op) == "tt.store").operands[1].owner.name == "arith.subi"


def test_addptr_increment_wraps_at_its_source_width_before_widening_to_bytes():
    module = _module('''%short=arith.trunci %offset:i64 to i32
      %two=arith.constant 2:i32
      %increment=arith.muli %short,%two:i32
      %ptr="tt.addptr"(%p,%increment):(!tt.ptr<i32>,i32)->!tt.ptr<i32>
      %address="tt.ptr_to_int"(%ptr):(!tt.ptr<i32>)->i64
      %delta=arith.subi %address,%bits:i64
      "tt.store"(%out,%delta):(!tt.ptr<i64>,i64)->()''')
    original = next(op for op in module.walk() if op.name == "arith.muli")
    normalize_pointer_addresses(module, _source_ids)
    module.verify()
    extensions = [op for op in module.walk() if isinstance(op, arith.ExtSIOp)]
    assert len(extensions) == 1 and extensions[0].operands[0] is original.results[0]
    assert original.operands[0].type == builtin.i32 and original.parent is not None
    assert any(op.name == "arith.muli" and op.operands[0] is extensions[0].result for op in module.walk())


@pytest.mark.parametrize("body,message", [
    ('''%other="tt.ptr_to_int"(%out) {prov.root="same"}:(!tt.ptr<i64>)->i64
        %delta=arith.subi %bits,%other:i64''', "same established logical base"),
    ('''%first=arith.addi %bits,%offset:i64
        %delta=arith.subi %first,%bits:i64
        "tt.store"(%out,%bits):(!tt.ptr<i64>,i64)->()''', "observe or escape"),
    ('''%delta="arith.subi"(%bits,%bits)<{overflowFlags=#arith.overflow<nsw>}>:(i64,i64)->i64''', "non-poisoning"),
    ('''%delta=arith.subi %bits,%bits:i64
        %ptr="tt.int_to_ptr"(%delta):(i64)->!tt.ptr<i32>''', "original pointer base"),
    ('''%ptr="tt.addptr"(%p,%offset) {unknown_address_policy,prov.family="pure"}:(!tt.ptr<i32>,i64)->!tt.ptr<i32>
        %address="tt.ptr_to_int"(%ptr):(!tt.ptr<i32>)->i64
        %delta=arith.subi %address,%bits:i64''', "unqualified source fields"),
])
def test_unproved_roots_observations_poison_and_source_fields_refuse_transactionally(body, message):
    unchanged_refusal(_module(body), message)


def test_nonuniform_pointer_lanes_cannot_cancel_through_an_unmatched_permutation():
    module = parse_shared('''module {"tt.func"() <{sym_name="lanes",
      function_type=(tensor<2x2x!tt.ptr<i32>>)->()}> ({
      ^bb0(%p:tensor<2x2x!tt.ptr<i32>>):
        %trans="tt.trans"(%p)<{order=array<i32:1,0>}>:
          (tensor<2x2x!tt.ptr<i32>>)->tensor<2x2x!tt.ptr<i32>>
        %first="tt.ptr_to_int"(%trans):(tensor<2x2x!tt.ptr<i32>>)->tensor<2x2xi64>
        %second="tt.ptr_to_int"(%p):(tensor<2x2x!tt.ptr<i32>>)->tensor<2x2xi64>
        %delta=arith.subi %first,%second:tensor<2x2xi64>
        "tt.return"():()->()
      }):()->() }''')
    unchanged_refusal(module, "same established logical base and lane correspondence")


@pytest.mark.parametrize("serialized", [False, True])
def test_matching_nonuniform_lane_paths_ignore_provenance_and_field_order(serialized):
    module = parse_shared('''module {"tt.func"() <{sym_name="lanes",
      function_type=(tensor<2x2x!tt.ptr<i32>>)->()}> ({
      ^bb0(%p:tensor<2x2x!tt.ptr<i32>>):
        %a="tt.trans"(%p)<{order=array<i32:1,0>,prov.origin="first"}>:
          (tensor<2x2x!tt.ptr<i32>>)->tensor<2x2x!tt.ptr<i32>>
        %b="tt.trans"(%p)<{prov.origin="second",order=array<i32:1,0>}>:
          (tensor<2x2x!tt.ptr<i32>>)->tensor<2x2x!tt.ptr<i32>>
        %first="tt.ptr_to_int"(%a):(tensor<2x2x!tt.ptr<i32>>)->tensor<2x2xi64>
        %second="tt.ptr_to_int"(%b):(tensor<2x2x!tt.ptr<i32>>)->tensor<2x2xi64>
        %delta=arith.subi %first,%second:tensor<2x2xi64>
        "tt.return"():()->()
      }):()->() }''')
    if serialized:
        module = parse_shared(text(module))
    reports = normalize_pointer_addresses(module, _source_ids)
    module.verify()
    assert len(reports) == 1 and reports[0]["kind"] == "closed-pointer-integer-difference"


def test_same_base_cancellation_is_independent_of_physical_address_and_unsigned_wrap():
    mask = (1 << 64) - 1
    offsets = (0, 1, -1, -(1 << 63), (1 << 63) - 1, 1 << 62, -129)
    for base in (0, 1, mask, mask - 3, 1 << 63):
        for first in offsets:
            for second in offsets:
                address_difference = (((base + first) & mask) - ((base + second) & mask)) & mask
                assert address_difference == (first - second) & mask


def test_pointer_recipe_analysis_has_a_bounded_source_path():
    parts = ["%p0=\"tt.addptr\"(%p,%offset):(!tt.ptr<i32>,i64)->!tt.ptr<i32>"]
    parts += [f'%p{i+1}="tt.addptr"(%p{i},%offset):(!tt.ptr<i32>,i64)->!tt.ptr<i32>' for i in range(130)]
    parts += ['%address="tt.ptr_to_int"(%p130):(!tt.ptr<i32>)->i64', '%delta=arith.subi %address,%bits:i64']
    unchanged_refusal(_module("\n".join(parts)), "bounded SSA budget")
