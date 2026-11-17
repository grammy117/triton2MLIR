"""Rebased seed recipes retain source units, actual roots and complete rollback."""

import pytest
from test_integer_address_region_contract import unchanged_refusal
from xdsl.dialects import arith, builtin

from t2m.ir import text
from t2m.pointer_addresses import normalize_pointer_addresses
from t2m.shared_tensor import _source_ids, opcode, parse_shared


def seed_source(width=32, *, tiled=False):
    if tiled:
        return f'''module {{"tt.func"() <{{sym_name="seed_tile",
          function_type=(!tt.ptr<i{width}>,!tt.ptr<i{width}>,!tt.ptr<i64>,i64,i32)->()}}> ({{
          ^bb0(%p:!tt.ptr<i{width}>,%out:!tt.ptr<i{width}>,%dist:!tt.ptr<i64>,%count:i64,%capacity:i32):
            %zero=arith.constant 0:i64
            %lanes="tt.make_range"() <{{start=0:i32,end=4:i32}}>:()->tensor<4xi32>
            %base="tt.splat"(%p):(!tt.ptr<i{width}>)->tensor<4x!tt.ptr<i{width}>>
            %bits="tt.ptr_to_int"(%base):(tensor<4x!tt.ptr<i{width}>>)->tensor<4xi64>
            %outbase="tt.splat"(%out):(!tt.ptr<i{width}>)->tensor<4x!tt.ptr<i{width}>>
            %outptr="tt.addptr"(%outbase,%lanes):(tensor<4x!tt.ptr<i{width}>>,tensor<4xi32>)->tensor<4x!tt.ptr<i{width}>>
            %result="scf.while"(%count) ({{
            ^bb1(%before:i64):
              %condition=arith.cmpi sgt,%before,%zero:i64
              %short=arith.trunci %before:i64 to i32
              %offset="tt.splat"(%short):(i32)->tensor<4xi32>
              %increment=arith.addi %offset,%lanes:tensor<4xi32>
              %local="tt.addptr"(%base,%increment):(tensor<4x!tt.ptr<i{width}>>,tensor<4xi32>)->tensor<4x!tt.ptr<i{width}>> loc("ttir.op.local")
              %address="tt.ptr_to_int"(%local):(tensor<4x!tt.ptr<i{width}>>)->tensor<4xi64> loc("ttir.op.seed")
              "scf.condition"(%condition,%address):(i1,tensor<4xi64>)->()
            }},{{
            ^bb2(%after:tensor<4xi64>):
              %pointer="tt.int_to_ptr"(%after):(tensor<4xi64>)->tensor<4x!tt.ptr<i{width}>> loc("ttir.op.pointer")
              %loaded="tt.load"(%pointer) <{{operandSegmentSizes=array<i32:1,0,0>}}>:(tensor<4x!tt.ptr<i{width}>>)->tensor<4xi{width}>
              "tt.store"(%outptr,%loaded) <{{operandSegmentSizes=array<i32:1,1,0>}}>:(tensor<4x!tt.ptr<i{width}>>,tensor<4xi{width}>)->()
              "scf.yield"(%zero):(i64)->()
            }}):(i64)->tensor<4xi64> loc("ttir.op.while")
            %delta=arith.subi %result,%bits:tensor<4xi64> loc("ttir.op.difference")
            %distbase="tt.splat"(%dist):(!tt.ptr<i64>)->tensor<4x!tt.ptr<i64>>
            %distptr="tt.addptr"(%distbase,%lanes):(tensor<4x!tt.ptr<i64>>,tensor<4xi32>)->tensor<4x!tt.ptr<i64>>
            "tt.store"(%distptr,%delta) <{{operandSegmentSizes=array<i32:1,1,0>}}>:(tensor<4x!tt.ptr<i64>>,tensor<4xi64>)->()
            "tt.return"():()->()
          }}):()->() }}'''
    widen = f'%wide=arith.extsi %loaded:i{width} to i64' if width != 64 else '%wide=arith.addi %loaded,%zero:i64'
    return f'''module {{"tt.func"() <{{sym_name="seed_scalar",
      function_type=(!tt.ptr<i{width}>,!tt.ptr<i64>,i64,i32)->()}}> ({{
      ^bb0(%p:!tt.ptr<i{width}>,%out:!tt.ptr<i64>,%count:i64,%capacity:i32):
        %zero=arith.constant 0:i64
        %three=arith.constant 3:i32
        %five=arith.constant 5:i32
        %bits="tt.ptr_to_int"(%p):(!tt.ptr<i{width}>)->i64
        %result="scf.while"(%count) ({{
        ^bb1(%before:i64):
          %condition=arith.cmpi sgt,%before,%zero:i64
          %short=arith.trunci %before:i64 to i32
          %scaled=arith.muli %short,%three:i32
          %increment=arith.addi %scaled,%five:i32
          %local="tt.addptr"(%p,%increment):(!tt.ptr<i{width}>,i32)->!tt.ptr<i{width}> loc("ttir.op.local")
          %address="tt.ptr_to_int"(%local):(!tt.ptr<i{width}>)->i64 loc("ttir.op.seed")
          "scf.condition"(%condition,%address):(i1,i64)->()
        }},{{
        ^bb2(%after:i64):
          %pointer="tt.int_to_ptr"(%after):(i64)->!tt.ptr<i{width}> loc("ttir.op.pointer")
          %loaded="tt.load"(%pointer) <{{operandSegmentSizes=array<i32:1,0,0>}}>:( !tt.ptr<i{width}>)->i{width}
          {widen}
          "scf.yield"(%wide):(i64)->()
        }}):(i64)->i64 loc("ttir.op.while")
        %delta=arith.subi %result,%bits:i64 loc("ttir.op.difference")
        "tt.store"(%out,%delta) <{{operandSegmentSizes=array<i32:1,1,0>}}>:( !tt.ptr<i64>,i64)->()
        "tt.return"():()->()
      }}):()->() }}'''


def unaligned_source():
    # A true byte view crosses a word boundary. The original typed pointer can
    # remain a base without asserting that its displacement from p is aligned.
    return seed_source().replace(
        '%local="tt.addptr"(%p,%increment):(!tt.ptr<i32>,i32)->!tt.ptr<i32>',
        '''%byte="tt.bitcast"(%p):(!tt.ptr<i32>)->!tt.ptr<i8>
          %one=arith.constant 1:i32
          %unaligned="tt.addptr"(%byte,%one):(!tt.ptr<i8>,i32)->!tt.ptr<i8>
          %local="tt.bitcast"(%unaligned):(!tt.ptr<i8>)->!tt.ptr<i32>''').replace(
        '%result="scf.while"', '%result:2="scf.while"').replace(
        '"scf.condition"(%condition,%address):(i1,i64)',
        '"scf.condition"(%condition,%address,%before):(i1,i64,i64)').replace(
        '^bb2(%after:i64)', '^bb2(%after:i64,%previous:i64)').replace(
        '}):(i64)->i64 loc("ttir.op.while")', '}):(i64)->(i64,i64) loc("ttir.op.while")').replace(
        '%delta=arith.subi %result,%bits:i64', '%delta=arith.addi %result#1,%zero:i64')


@pytest.mark.parametrize("tiled", [False, True])
def test_rebased_seed_keeps_original_narrow_computation_and_real_shaped_anchor(tiled):
    module = parse_shared(seed_source(tiled=tiled))
    reports = normalize_pointer_addresses(module, _source_ids)
    if not tiled:
        module.verify()
    assert any(item["kind"] == "integer-address-seed-rebase" for item in reports)
    assert not any(opcode(op) in ("tt.ptr_to_int", "tt.int_to_ptr") for op in module.walk())
    loop = next(op for op in module.walk() if op.name == "scf.while")
    # Native MLIR verifies the shaped standard extsi in the end-to-end test;
    # xDSL 0.68's extsi binding only accepts scalars. Check the actual region
    # interface here before the official shaped-arithmetic conversion stage.
    loop.verify(verify_nested_ops=False)
    assert len(loop.operands) == len(loop.regions[0].block.args) == 1
    assert len(loop.results) == len(loop.regions[1].block.args) == 2
    assert any(opcode(op) == "tt.load" for op in loop.walk())
    if tiled:
        anchor = loop.regions[0].block.last_op.operands[2]
        assert anchor.owner.parent is loop.parent and opcode(anchor.owner) == "tt.splat"
    if not tiled:
        multiply = next(op for op in loop.walk() if op.name == "arith.muli" and op.results[0].type == builtin.i32)
        assert multiply.operands[0].owner.name == "arith.trunci"
        extension = next(op for op in loop.walk() if isinstance(op, arith.ExtSIOp)
                         and op.operands[0].owner.name == "arith.addi" and op.operands[0].type == builtin.i32)
        assert extension.operands[0].owner.operands[0] is multiply.results[0]


def test_rebased_byte_offsets_do_not_inherit_the_old_seed_zero_displacement_fact():
    source = seed_source().replace(
        '%local="tt.addptr"(%p,%increment):(!tt.ptr<i32>,i32)->!tt.ptr<i32>',
        '''%byte="tt.bitcast"(%p):(!tt.ptr<i32>)->!tt.ptr<i8>
          %one=arith.constant 1:i32
          %unaligned="tt.addptr"(%byte,%one):(!tt.ptr<i8>,i32)->!tt.ptr<i8>
          %local="tt.bitcast"(%unaligned) {prov.alignment=8:i64}:(!tt.ptr<i8>)->!tt.ptr<i32>''')
    unchanged_refusal(parse_shared(source), "not proved divisible")


def test_existing_unaligned_pointer_representation_remains_available_without_a_false_bit_fact():
    module = parse_shared(unaligned_source())
    reports = normalize_pointer_addresses(module, _source_ids)
    module.verify()
    assert any(item["kind"] == "integer-address-region-state" for item in reports)
    assert not any(item["kind"] == "integer-address-seed-rebase" for item in reports)
    assert not any(opcode(op) in ("tt.ptr_to_int", "tt.int_to_ptr") for op in module.walk())


@pytest.mark.parametrize("serialized", [False, True])
def test_unaligned_seed_retention_does_not_disable_a_different_required_root_rebase(serialized):
    module = parse_shared(unaligned_source())
    other = parse_shared(seed_source().replace('sym_name="seed_scalar"', 'sym_name="aligned"'))
    function = other.body.block.detach_op(other.body.block.first_op)
    module.body.block.add_op(function)
    if serialized:
        module = parse_shared(text(module))
    reports = normalize_pointer_addresses(module, _source_ids)
    module.verify()
    assert any(item["kind"] == "integer-address-seed-rebase" for item in reports)
    assert any(item["kind"] == "integer-address-seed-retention" for item in reports)
    assert not any(opcode(op) in ("tt.ptr_to_int", "tt.int_to_ptr") for op in module.walk())


def test_optional_seed_retention_has_a_bounded_transactional_refinement_budget():
    module = parse_shared(unaligned_source())
    for index in range(16):
        other = parse_shared(unaligned_source().replace('sym_name="seed_scalar"', f'sym_name="seed{index}"'))
        module.body.block.add_op(other.body.block.detach_op(other.body.block.first_op))
    unchanged_refusal(module, "16-refinement budget")


@pytest.mark.parametrize("change,message", [
    ('%address="tt.ptr_to_int"(%local) {unknown_policy,prov.family="proved"}', "unqualified source fields"),
    ('%address="tt.ptr_to_int"(%local)', "observe or escape"),
])
def test_rebased_source_fields_and_late_absolute_observations_leave_original_uses_unchanged(change, message):
    source = seed_source().replace('%address="tt.ptr_to_int"(%local)', change)
    if message == "observe or escape":
        source = source.replace('"scf.condition"(%condition,%address)',
            '"tt.store"(%out,%address):(!tt.ptr<i64>,i64)->()\n "scf.condition"(%condition,%address)')
    unchanged_refusal(parse_shared(source), message)
