"""Actual step identity and exit-state edges authorize pointer re-raising."""

import pytest

from t2m.loop_domains import enclosed_loop, is_exit_induction, is_iteration_ordinal
from t2m.shared_tensor import _source_ids, opcode, parse_shared
from t2m.source_pointer_recurrences import raise_pointer_recurrences
from t2m.structured_state import data_control_report


def source(*, increment="%delta", initial="%lo", exit_update="%next_iv"):
    return parse_shared(f'''builtin.module {{
      "tt.func"() <{{sym_name="stream", function_type=(!tt.ptr<i64>, i32, i32, i32, i32) -> ()}}> ({{
      ^bb0(%ptr: !tt.ptr<i64>, %lo: i32, %hi: i32, %step: i32, %other: i32):
        %two = arith.constant 2 : i32
        %delta = arith.muli %step, %two : i32
        %zero = arith.constant dense<0> : tensor<4xi64>
        %result:3 = scf.for %iv = %lo to %hi step %step
            iter_args(%p = %ptr, %data = %zero, %exit = {initial}) -> (!tt.ptr<i64>, tensor<4xi64>, i32) : i32 {{
          %value = "tt.load"(%p) : (!tt.ptr<i64>) -> i64
          %next = "tt.addptr"(%p, {increment}) : (!tt.ptr<i64>, i32) -> !tt.ptr<i64>
          %next_data = arith.addi %data, %zero : tensor<4xi64>
          %next_iv = arith.addi %iv, %step : i32
          %wrong_iv = arith.addi %iv, %other : i32
          scf.yield %next, %next_data, {exit_update} : !tt.ptr<i64>, tensor<4xi64>, i32
        }} loc("ttir.op.loop")
        %last = "tt.load"(%result#0) : (!tt.ptr<i64>) -> i64
        "tt.return"() : () -> ()
      }}) : () -> ()
    }}''')


def test_actual_step_normalization_retains_numeric_state_memory_and_observed_exit():
    module = source()
    loop = next(op for op in module.walk() if op.name == "scf.for")
    report = raise_pointer_recurrences(module, _source_ids,
        {loop: enclosed_loop(((0, 3), (0, 8), (1, 3)), width=32)})
    assert report[0]["streams"][0]["step_coefficient"] == 2
    loop = next(op for op in module.walk() if op.name == "scf.for")
    assert [str(v.type) for v in loop.results] == ["tensor<4xi64>", "i32", "i32"]
    assert is_exit_induction(loop, 1) and is_exit_induction(loop, 2)
    assert sum(opcode(op) == "tt.load" for op in module.walk()) == 2
    assert data_control_report(module, _source_ids)["upstream_option"] == "skip-prepass=true"
    module.verify()


@pytest.mark.parametrize("increment", ["%other", "%iv"])
def test_equal_declared_ranges_or_unrelated_increments_supply_no_step_equality(increment):
    module = source(increment=increment)
    loop = next(op for op in module.walk() if op.name == "scf.for")
    before = str(module)
    assert raise_pointer_recurrences(module, _source_ids,
        {loop: enclosed_loop(((0, 3), (0, 8), (1, 3)), width=32)}) == []
    assert str(module) == before


@pytest.mark.parametrize("initial,update,expected", [
    ("%lo", "%next_iv", True), ("%other", "%next_iv", False), ("%lo", "%wrong_iv", False),
])
def test_exit_fact_requires_the_original_initial_edge_and_actual_iv_step(initial, update, expected):
    module = source(initial=initial, exit_update=update)
    loop = next(op for op in module.walk() if op.name == "scf.for")
    loop.attributes["prov.exit_induction"] = loop.location
    assert is_exit_induction(loop, 2) is expected


def test_unproved_domains_do_not_rewrite_or_borrow_a_numeric_state_option():
    module = source()
    assert raise_pointer_recurrences(module, _source_ids, {}) == []
    assert data_control_report(module, _source_ids) is None


def test_invariant_pointer_keeps_its_actual_initial_value():
    module = source()
    loop = next(op for op in module.walk() if op.name == "scf.for")
    end = loop.body.block.last_op
    end.operands = [loop.body.block.args[1], *end.operands[1:]]
    report = raise_pointer_recurrences(module, _source_ids,
        {loop: enclosed_loop(((0, 3), (0, 8), (1, 3)), width=32)})
    assert report[0]["streams"][0]["step_coefficient"] == 0
    last = next(op for op in module.ops.first.regions[0].block.ops if opcode(op) == "tt.load")
    assert last.operands[0] is module.ops.first.regions[0].block.args[0]


@pytest.mark.parametrize("increment,expected", [("%two", (0, 2)), ("%mixed", (2, 2))])
def test_constant_and_mixed_streams_keep_a_real_unit_increment_ordinal(increment, expected):
    module = source(increment=increment.replace("%mixed", "%two"))
    loop = next(op for op in module.walk() if op.name == "scf.for")
    if increment == "%mixed":
        from xdsl.dialects import arith
        update = loop.body.block.last_op.operands[0].owner
        two = update.operands[1]
        delta = next(op for op in loop.parent.ops if op.name == "arith.muli").result
        mixed = arith.AddiOp(delta, two)
        loop.parent.insert_op_before(mixed, loop)
        update.operands = [update.operands[0], mixed.result]
    report = raise_pointer_recurrences(module, _source_ids,
        {loop: enclosed_loop(((0, 3), (0, 8), (1, 3)), width=32)})
    stream = report[0]["streams"][0]
    assert report[0]["kind"] == "source-ordinal-pointer-recurrences"
    assert (stream["step_coefficient"], stream["iteration_increment"]) == expected
    loop = next(op for op in module.walk() if op.name == "scf.for")
    assert is_iteration_ordinal(loop, len(loop.results)-1)
    assert data_control_report(module, _source_ids)
    module.verify()
