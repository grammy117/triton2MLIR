"""Invariant means actual SSA dependence, with no speculation or width grant."""

import pytest
from test_source_pointer_recurrences import source

from t2m.diagnostics import ConversionError
from t2m.invariant_recurrences import (
    invariant_increment,
    is_loop_invariant,
    recurrence_bounds,
)
from t2m.loop_domains import enclosed_loop
from t2m.shared_tensor import _source_ids, opcode
from t2m.source_pointer_recurrences import raise_pointer_recurrences


def loop_and_ranges(module):
    loop = next(op for op in module.walk() if op.name == "scf.for")
    args = module.ops.first.regions[0].block.args
    return loop, {args[3]: (1, 3), args[4]: (-2, 2)}


@pytest.mark.parametrize("headers", [((0, 3), (0, 8), (1, 3)), ((0, 0), (8, 8), (1, 1))])
def test_unrelated_runtime_stride_keeps_its_own_actual_scalar_and_axis_factor(headers):
    module = source(increment="%other")
    loop, ranges = loop_and_ranges(module)
    # Equal declared intervals do not turn this different SSA value into step.
    ranges[module.ops.first.regions[0].block.args[4]] = (1, 3)
    report = raise_pointer_recurrences(module, _source_ids,
        {loop: enclosed_loop(headers, width=32)}, scalar_ranges=ranges)
    assert report[0]["kind"] == "source-invariant-runtime-pointer-recurrences"
    assert report[0]["streams"] == [{"carried_argument": 0, "step_coefficient": 0,
        "logical_unit_factor": 1, "invariant_scalar_terms": 1}]
    # Original numeric tuple and both actual memory observations survive.
    loop = next(op for op in module.walk() if op.name == "scf.for")
    assert str(loop.iter_args[0].type) == "tensor<4xi64>"
    assert sum(opcode(op) == "tt.load" for op in module.walk()) == 2
    assert invariant_increment(loop, len(loop.iter_args)-1, opcode)[1] == 1
    module.verify()


@pytest.mark.parametrize("increment", ["%iv", "%value"])
def test_varying_iv_or_body_load_needs_its_actual_enclosure_and_never_claims_invariance(increment):
    from xdsl.dialects import builtin

    module = source(increment="%iv")
    loop, ranges = loop_and_ranges(module)
    update = loop.body.block.last_op.operands[0].owner
    value = loop.body.block.args[0] if increment == "%iv" else next(op for op in loop.body.block.ops if opcode(op) == "tt.load").results[0]
    update.operands = [update.operands[0], value]
    loop.attributes["prov.invariant"] = builtin.StringAttr("claimed")
    before = str(module)
    assert not is_loop_invariant(value, loop, opcode)
    assert raise_pointer_recurrences(module, _source_ids,
        {loop: enclosed_loop(((0, 3), (0, 8), (1, 3)), width=32)}, scalar_ranges=ranges) == []
    assert str(module) == before
    ranges[value] = (0, 7) if increment == "%iv" else (-(1 << 63), (1 << 63)-1)
    reports = raise_pointer_recurrences(module, _source_ids,
        {loop: enclosed_loop(((0, 3), (0, 8), (1, 3)), width=32)}, scalar_ranges=ranges)
    if increment == "%iv":
        assert reports[0]["kind"] == "source-bounded-runtime-pointer-recurrences"
        assert reports[0]["streams"][0]["bounded_scalar_terms"] == 1
        assert "invariant_scalar_terms" not in reports[0]["streams"][0]
        module.verify()
    else:
        assert reports == [] and str(module) == before


def test_body_defined_division_stays_inside_actual_loop_and_uses_original_definition():
    from xdsl.dialects import arith

    module = source(increment="%other")
    loop, ranges = loop_and_ranges(module)
    quotient = arith.DivSIOp(loop.parent.args[4], loop.ub)
    loop.body.block.insert_op_before(quotient, loop.body.block.first_op)
    update = loop.body.block.last_op.operands[0].owner
    update.operands = [update.operands[0], quotient.result]
    ranges[quotient.result] = (-2, 2)
    report = raise_pointer_recurrences(module, _source_ids,
        {loop: enclosed_loop(((0, 3), (0, 8), (1, 3)), width=32)}, scalar_ranges=ranges)
    assert report
    division = next(op for op in module.walk() if op.name == "arith.divsi")
    loop = next(op for op in module.walk() if op.name == "scf.for")
    assert division.parent is loop.body.block
    assert sum(op.name == "arith.divsi" for op in module.walk()) == 1
    module.verify()


def test_signed_progress_encloses_both_directions_and_original_zero_trip_initials():
    domain = enclosed_loop(((0, 3), (0, 8), (1, 3)), width=32)
    assert recurrence_bounds(domain, (16, 19), (-2, 2), width=32) == ((2, 33), (0, 35))


def test_small_carrier_overflow_refuses_even_when_the_header_width_is_valid():
    domain = enclosed_loop(((0, 0), (0, 8), (1, 1)), width=32)
    with pytest.raises(ConversionError, match="carried width"):
        recurrence_bounds(domain, (120, 120), (1, 2), width=8)


def test_unrepresentable_scaled_runtime_offset_does_not_become_logical_indices():
    from xdsl.dialects import arith

    module = source(increment="%other")
    loop, ranges = loop_and_ranges(module)
    other = loop.parent.args[4]
    huge = arith.ConstantOp.from_int_and_width((1 << 63)-1, 64)
    ext = arith.ExtSIOp(other, huge.result.type)
    scaled = arith.MuliOp(ext, huge)
    combined = arith.AddiOp(scaled, ext)
    loop.parent.insert_ops_before([huge, ext, scaled, combined], loop)
    update = loop.body.block.last_op.operands[0].owner
    update.operands = [update.operands[0], combined.result]
    ranges[other] = (1, 2)
    before = str(module)
    assert raise_pointer_recurrences(module, _source_ids,
        {loop: enclosed_loop(((0, 3), (0, 8), (1, 3)), width=32)}, scalar_ranges=ranges) == []
    assert str(module) == before


@pytest.mark.parametrize("increment,valid", [((-2, 2), True), ((-16, 16), False)])
def test_source_header_and_actual_memory_carrier_retain_independent_widths(increment, valid):
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_indexing import check_source_addresses
    from t2m.shared_tensor import TensorABI, TensorBuffer, parse_shared

    module = parse_shared('''builtin.module {
      "tt.func"() <{sym_name="index_stream", function_type=(!tt.ptr<i64>, i32, i32, i32, i8) -> ()}> ({
      ^bb0(%x: !tt.ptr<i64>, %lo: i32, %hi: i32, %step: i32, %amount: i8):
        %initial = arith.constant 16 : i8
        %end = scf.for %iv = %lo to %hi step %step iter_args(%position = %initial) -> (i8) : i32 {
          %wide = arith.extsi %position : i8 to i32
          %p = "tt.addptr"(%x, %wide) : (!tt.ptr<i64>, i32) -> !tt.ptr<i64>
          %value = "tt.load"(%p) : (!tt.ptr<i64>) -> i64
          %next = arith.addi %position, %amount : i8
          scf.yield %next : i8
        }
        %wide_end = arith.extsi %end : i8 to i32
        %last = "tt.addptr"(%x, %wide_end) : (!tt.ptr<i64>, i32) -> !tt.ptr<i64>
        %value = "tt.load"(%last) : (!tt.ptr<i64>) -> i64
        "tt.return"() : () -> ()
      }) : () -> ()
    }''')
    abi = TensorABI("index_stream", (TensorBuffer("x", (64,), "read", "i64"),), effect_only=True,
        runtime_scalars=tuple(RuntimeScalar(name, dtype, i, bounds=bounds) for i, (name, dtype, bounds) in enumerate(
            (("lo", "i32", (200, 203)), ("hi", "i32", (200, 208)), ("step", "i32", (1, 3)),
             ("amount", "i8", increment)), 1)))
    if not valid:
        with pytest.raises(ConversionError, match="carried width"):
            check_source_addresses(module, abi)
        return
    ranges = {}
    check_source_addresses(module, abi, scalar_ranges=ranges)
    loop = next(op for op in module.walk() if op.name == "scf.for")
    assert ranges[loop.body.block.args[1]] == (2, 30)
    assert ranges[loop.results[0]] == (0, 32)
