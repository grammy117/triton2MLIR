"""Address region state needs executable edges, bit facts and complete rollback."""

from io import StringIO

import pytest
from xdsl.printer import Printer

from t2m.diagnostics import ConversionError
from t2m.ir import text
from t2m.pointer_addresses import normalize_pointer_addresses
from t2m.shared_tensor import _source_ids, opcode, parse_shared


def loop_source(body="", *, increment="%four", bound="%steps", tail="", poison=False):
    update = ('%next="arith.addi"(%address,' + increment + ') <{overflowFlags=#arith.overflow<nsw>}>:(i64,i64)->i64'
              if poison else f'%next=arith.addi %address,{increment}:i64')
    update += ' loc("ttir.op.next")'
    return f'''module {{"tt.func"() <{{sym_name="region",
      function_type=(!tt.ptr<i32>,!tt.ptr<i64>,i64,i1)->()}}> ({{
      ^bb0(%p:!tt.ptr<i32>,%out:!tt.ptr<i64>,%steps:i64,%choice:i1):
        %bits="tt.ptr_to_int"(%p):(!tt.ptr<i32>)->i64 loc("ttir.op.base")
        %zero=arith.constant 0:i64
        %one=arith.constant 1:i64
        %four=arith.constant 4:i64
        %result="scf.for"(%zero,{bound},%one,%bits) ({{
        ^bb1(%i:i64,%address:i64):
          %pointer="tt.int_to_ptr"(%address) {{prov.alignment=8:i64}}:(i64)->!tt.ptr<i32> loc("ttir.op.pointer")
          {body}
          {update}
          "scf.yield"(%next):(i64)->()
        }}):(i64,i64,i64,i64)->i64 loc("ttir.op.loop")
        %delta=arith.subi %result,%bits:i64 loc("ttir.op.difference")
        "tt.store"(%out,%delta):(!tt.ptr<i64>,i64)->()
        {tail}
        "tt.return"():()->()
      }}):()->() }}'''


def unchanged_refusal(module, message):
    before = text(module)
    values = [value for op in module.walk() for value in (
        *op.results, *(arg for region in op.regions for block in region.blocks for arg in block.args))]
    uses = {value: set(value.uses) for value in values}
    with pytest.raises(ConversionError, match=message) as error:
        normalize_pointer_addresses(module, _source_ids)
    assert error.value.opcode and error.value.source_id
    assert text(module) == before and {value: set(value.uses) for value in values} == uses


def test_for_keeps_original_integer_slots_and_effects_with_internal_pointer_fields():
    module = parse_shared(loop_source('%loaded="tt.load"(%pointer):(!tt.ptr<i32>)->i32'))
    reports = normalize_pointer_addresses(module, _source_ids)
    module.verify()
    loop = next(op for op in module.walk() if op.name == "scf.for")
    assert [str(value.type) for value in loop.regions[0].block.args] == ["i64", "i64", "!tt.ptr<i32>"]
    assert len(loop.results) == 2 and len(loop.operands) == 5
    assert any(opcode(op) == "tt.load" for op in loop.walk())
    assert not any(opcode(op) in ("tt.ptr_to_int", "tt.int_to_ptr") for op in module.walk())
    assert any(report["kind"] == "integer-address-region-state" for report in reports)


@pytest.mark.parametrize("source,message", [
    (loop_source('"tt.store"(%out,%address):(!tt.ptr<i64>,i64)->()'), "observe or escape"),
    (loop_source('%short=arith.trunci %address:i64 to i32'), "observe or escape"),
    (loop_source(increment="%one"), "divisible"),
    (loop_source(bound="%bits"), "control region bounds"),
    (loop_source(poison=True), "non-poisoning"),
    (loop_source(tail='''%other="tt.ptr_to_int"(%out):(!tt.ptr<i64>)->i64
      %escape=arith.subi %result,%other:i64 loc("ttir.op.other-root")'''), "same established logical base"),
])
def test_no_partial_pairing_or_bit_fact_escapes_a_later_refusal(source, message):
    unchanged_refusal(parse_shared(source), message)


@pytest.mark.parametrize("serialized", [False, True])
def test_numeric_or_address_phi_mixture_cannot_be_authorized_by_provenance(serialized):
    source = loop_source().replace('%next=arith.addi %address,%four:i64',
        '%next=arith.select %choice,%address,%zero {prov.address_state="proved"}:i64')
    module = parse_shared(source)
    if serialized:
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(module)
        module = parse_shared(stream.getvalue())
    unchanged_refusal(module, "closed address-only state")


def while_source():
    return '''module {"tt.func"() <{sym_name="while_ports",
      function_type=(!tt.ptr<i32>,!tt.ptr<i64>,i64)->()}> ({
      ^bb0(%p:!tt.ptr<i32>,%out:!tt.ptr<i64>,%count:i64):
        %zero=arith.constant 0:i64
        %bits="tt.ptr_to_int"(%p):(!tt.ptr<i32>)->i64
        %result="scf.while"(%count) ({
        ^bb1(%before:i64):
          %condition=arith.cmpi sgt,%before,%zero:i64
          %address="tt.ptr_to_int"(%p):(!tt.ptr<i32>)->i64
          "scf.condition"(%condition,%address):(i1,i64)->()
        },{
        ^bb2(%after:i64):
          %pointer="tt.int_to_ptr"(%after):(i64)->!tt.ptr<i32>
          %loaded="tt.load"(%pointer) <{operandSegmentSizes=array<i32:1,0,0>}>:(!tt.ptr<i32>)->i32
          %wide=arith.extsi %loaded:i32 to i64
          "scf.yield"(%wide):(i64)->()
        }):(i64)->i64 loc("ttir.op.while")
        %delta=arith.subi %result,%bits:i64
        "tt.store"(%out,%delta) <{operandSegmentSizes=array<i32:1,1,0>}>:(!tt.ptr<i64>,i64)->()
        "tt.return"():()->()
      }):()->() }'''


def test_while_preserves_different_original_input_and_output_arities():
    module = parse_shared(while_source())
    normalize_pointer_addresses(module, _source_ids)
    module.verify()
    loop = next(op for op in module.walk() if op.name == "scf.while")
    assert len(loop.operands) == len(loop.regions[0].block.args) == 1
    assert len(loop.results) == len(loop.regions[1].block.args) == 2
    assert len(loop.regions[0].block.last_op.operands) == 3
    assert len(loop.regions[1].block.last_op.operands) == 1
    assert any(opcode(op) == "tt.load" for op in loop.walk())


def scoped_source():
    return '''module {"tt.func"() <{sym_name="scoped",
      function_type=(!tt.ptr<i32>,i64)->()}> ({
      ^bb0(%p:!tt.ptr<i32>,%count:i64):
        %zero=arith.constant 0:i64
        %bits="tt.ptr_to_int"(%p):(!tt.ptr<i32>)->i64
        %result="scf.while"(%count) ({
        ^bb1(%before:i64):
          %condition=arith.cmpi sgt,%before,%zero:i64
          %local="tt.addptr"(%p,%zero):(!tt.ptr<i32>,i64)->!tt.ptr<i32>
          %address="tt.ptr_to_int"(%local):(!tt.ptr<i32>)->i64
          "scf.condition"(%condition,%address):(i1,i64)->()
        },{
        ^bb2(%after:i64):
          "scf.yield"(%zero):(i64)->()
        }):(i64)->i64 loc("ttir.op.scope")
        %delta=arith.subi %result,%bits:i64 loc("ttir.op.scoped-difference")
        "tt.return"():()->()
      }):()->() }'''


def test_before_local_zero_offset_is_rebased_without_publishing_an_out_of_scope_value():
    module = parse_shared(scoped_source())
    reports = normalize_pointer_addresses(module, _source_ids)
    module.verify()
    assert any(item["kind"] == "integer-address-seed-rebase" for item in reports)
    assert not any(opcode(op) in ("tt.ptr_to_int", "tt.int_to_ptr") for op in module.walk())


def test_unproved_branch_local_base_is_not_published_as_an_out_of_scope_loop_invariant():
    source = scoped_source().replace('function_type=(!tt.ptr<i32>,i64)',
        'function_type=(!tt.ptr<i32>,!tt.ptr<i32>,i64)').replace(
        '^bb0(%p:!tt.ptr<i32>,%count:i64)', '^bb0(%p:!tt.ptr<i32>,%q:!tt.ptr<i32>,%count:i64)').replace(
        '%local="tt.addptr"(%p,%zero):(!tt.ptr<i32>,i64)->!tt.ptr<i32>',
        '%local=arith.select %condition,%p,%q:!tt.ptr<i32>')
    unchanged_refusal(parse_shared(source), "same established logical base")
