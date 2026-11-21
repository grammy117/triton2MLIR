"""Selected logical offsets keep actual regions, roots, widths and effects."""

import pytest

from t2m.diagnostics import ConversionError
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_indexing import check_source_addresses
from t2m.shared_tensor import TensorABI, TensorBuffer, _source_ids, opcode, parse_shared
from t2m.source_branch_offsets import expose_branch_offsets


def source(*, pointer=False, other_root=False, outside=False, wrong_guard=False):
    result_type = "!tt.ptr<i64>" if pointer else "i32"
    then_value, else_value = ("%a", "%b") if pointer else ("%left", "%right")
    division = "%eager" if outside else "%quotient"
    eager = "%eager = arith.divsi %one, %amount : i32" if outside else ""
    guard = "%unrelated" if wrong_guard else "%amount"
    module = parse_shared(f'''builtin.module {{
      "tt.func"() <{{sym_name="selected", function_type=(!tt.ptr<i64>, !tt.ptr<i64>, i32, i32) -> ()}}> ({{
      ^bb0(%x: !tt.ptr<i64>, %y: !tt.ptr<i64>, %amount: i32, %unrelated: i32):
        %zero = arith.constant 0 : i32
        %one = arith.constant 1 : i32
        %factor = arith.constant 17 : i32
        {eager}
        %condition = arith.cmpi sgt, {guard}, %zero : i32
        %selected = scf.if %condition -> ({result_type}) {{
          %quotient = arith.divsi %one, %amount : i32
          %left = arith.muli {division}, %factor : i32
          %a = "tt.addptr"(%x, %left) : (!tt.ptr<i64>, i32) -> !tt.ptr<i64>
          scf.yield {then_value} : {result_type}
        }} else {{
          %right = arith.muli %unrelated, %factor : i32
          %b = "tt.addptr"({'%y' if other_root else '%x'}, %right) : (!tt.ptr<i64>, i32) -> !tt.ptr<i64>
          scf.yield {else_value} : {result_type}
        }} loc("ttir.op.choice")
        "tt.return"() : () -> ()
      }}) : () -> ()
    }}''')
    abi = TensorABI("selected", (TensorBuffer("x", (256,), "read", "i64"), TensorBuffer("y", (256,), "read", "i64")),
        effect_only=True, runtime_scalars=(RuntimeScalar("amount", "i32", 2, bounds=(0, 3)),
                                         RuntimeScalar("unrelated", "i32", 3, bounds=(0, 3))))
    return module, abi


@pytest.mark.parametrize("pointer", [False, True])
def test_actual_selected_yields_retain_branch_local_division_and_axis_factor(pointer):
    module, abi = source(pointer=pointer)
    ranges = {}
    check_source_addresses(module, abi, scalar_ranges=ranges)
    original = next(op for op in module.walk() if op.name == "scf.if")
    if not pointer:
        check_source_addresses(module, abi, scalar_ranges=ranges, requested_values=original.results)
    blocks, condition = tuple(r.block for r in original.regions), original.cond
    division = next(op for op in module.walk() if op.name == "arith.divsi")
    reports = expose_branch_offsets(module, ranges, _source_ids)
    assert reports[0]["results"][0]["logical_unit_factor"] == 17
    actual = next(op for op in module.walk() if op.name == "scf.if")
    assert tuple(r.block for r in actual.regions) == blocks
    assert division.parent is actual.true_region.block
    assert actual.cond is condition
    assert sum(opcode(op) == "tt.addptr" for op in module.walk()) == (3 if pointer else 2)
    check_source_addresses(module, abi)
    module.verify()


def test_different_actual_pointer_roots_do_not_acquire_a_common_base_from_matching_shapes():
    module, abi = source(pointer=True, other_root=True)
    ranges = {}
    check_source_addresses(module, abi, scalar_ranges=ranges)
    before = str(module)
    assert expose_branch_offsets(module, ranges, _source_ids) == []
    assert str(module) == before


def test_a_shared_outside_pointer_prefix_stays_reused_without_flattening_its_axis():
    from xdsl.dialects import builtin

    module, abi = source(pointer=True)
    branch = next(op for op in module.walk() if op.name == "scf.if")
    base = module.ops.first.regions[0].block.args[0]
    one = next(op.result for op in branch.parent.ops if op.name == "arith.constant" and op.value.value.data == 1)
    prefix = builtin.UnregisteredOp.with_name("tt.addptr").create(operands=[base, one], result_types=[base.type])
    branch.parent.insert_op_before(prefix, branch)
    for region in branch.regions:
        pointer = next(op for op in region.block.ops if opcode(op) == "tt.addptr")
        pointer.operands = [prefix.results[0], pointer.operands[1]]
    ranges = {}
    check_source_addresses(module, abi, scalar_ranges=ranges)
    report = expose_branch_offsets(module, ranges, _source_ids)
    assert report[0]["results"][0]["logical_unit_factor"] == 17
    rebuilt = next(op for op in module.ops.first.regions[0].block.ops if opcode(op) == "tt.addptr" and op is not prefix)
    assert rebuilt.operands[0] is prefix.results[0]
    module.verify()


@pytest.mark.parametrize("changes", [{"outside": True}, {"wrong_guard": True}])
def test_later_or_unrelated_conditions_cannot_validate_an_original_unsafe_division(changes):
    module, abi = source(**changes)
    with pytest.raises(ConversionError, match="zero divisor"):
        check_source_addresses(module, abi)


def test_selected_offset_with_no_checked_scalar_range_declines_without_mutation():
    module, _abi = source(pointer=True)
    before = str(module)
    assert expose_branch_offsets(module, {}, _source_ids) == []
    assert str(module) == before
