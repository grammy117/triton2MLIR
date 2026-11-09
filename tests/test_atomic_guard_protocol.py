"""Native guard decisions agree with independent access-history rules."""

from itertools import product
import os

import numpy as np
import pytest

from t2m.memory_contract import guard_layout

NATIVE = os.environ.get("T2M_NATIVE_PYTHON")


@pytest.mark.parametrize("grid,words,cost", [((1,), 1, 16), ((2**30,), 1, 16),
    ((2**30 + 1,), 2, 24), ((65536, 65536), 2, 24), ((2**31 - 1, 2**31 - 1), 3, 32)])
def test_atomic_layout_reserves_flags_and_covers_the_complete_grid(grid, words, cost):
    assert guard_layout({"grid": grid}, atomics=True) == {
        "owner_words_per_element": words, "guard_storage_bytes_per_output_element": cost, "owner_fragment_bits": 30}


def histories():
    cases = [()]
    for foreign in (1, 1 << 30, 1 << 60):
        alphabet = tuple(product((0, foreign), range(3)))  # read, ordinary write, atomic
        cases.extend(sequence for count in range(1, 5) for sequence in product(alphabet, repeat=count))
    return cases


def permitted(history):
    # This declarative oracle uses participant identities and access categories,
    # independently of bit encodings, complemented tags or emitted operations.
    return (len({pid for pid, _ in history}) <= 1 or all(kind == 0 for _, kind in history)
            or all(kind == 2 for _, kind in history))


def batch_module(count):
    from xdsl.dialects import arith, builtin, cf, func, memref, scf, tensor
    from xdsl.dialects.linalg.ops import FillOp
    from xdsl.ir import Block, Region
    from xdsl.rewriter import Rewriter
    from t2m.guarded_memory import Access, guard_loops
    from t2m.guard_protocol import ATOMIC
    from t2m.shared_tensor import TensorBuffer
    from t2m.standard_bufferization import ToTensorOp
    type_ = builtin.TensorType(builtin.i64, [count, 4])
    entry = Block(arg_types=[type_, type_, builtin.TensorType(builtin.i64, [count])])
    def constant(n, type_=builtin.IndexType()):
        op = arith.ConstantOp.from_int_and_width(n, type_)
        entry.add_op(op)
        return op.result
    zero, one, stop = constant(0), constant(1), constant(count)
    zero64, true = constant(0, builtin.i64), constant(1, builtin.i1)
    stores = []
    for size, element in ((3, builtin.i64), (1, builtin.i64), (1, builtin.i1), (count, builtin.i1)):
        op = memref.AllocOp.build(operands=[[], []], result_types=[builtin.MemRefType(element, [size])])
        entry.add_op(op)
        stores.append(op.memref)
    owners, visited, accepted, results = stores
    outer = Block(arg_types=[builtin.IndexType()])
    outer.add_ops([FillOp([zero64], [owners], res=[]), FillOp([zero64], [visited], res=[]), FillOp([true], [accepted], res=[])])
    length = tensor.ExtractOp(entry.args[2], [outer.args[0]], builtin.i64)
    upper = arith.IndexCastOp(length.result, builtin.IndexType())
    outer.add_ops([length, upper])
    steps = Block(arg_types=[builtin.IndexType()])
    pid = tensor.ExtractOp(entry.args[0], [outer.args[0], steps.args[0]], builtin.i64)
    mode = tensor.ExtractOp(entry.args[1], [outer.args[0], steps.args[0]], builtin.i64)
    steps.add_ops([pid, mode])
    access = Access(TensorBuffer("state", (1,), "readwrite"), (1,), offsets=(0,), strides=(1,))
    extra = {("extent", "state"): constant(1, builtin.i64), ("owners", "state"): owners,
             ("visited", "state"): visited, ("program", None): pid.result}
    for action in range(3):
        literal = arith.ConstantOp.from_int_and_width(action, builtin.i64)
        selected = arith.CmpiOp(mode.result, literal.result, "eq")
        branch = Block()
        branch.add_ops(guard_loops(pid, access, extra, write=action != 0, atomic=action == 2,
                                   owner_words=3, protocol=ATOMIC))
        if action == 1:
            branch.add_ops(guard_loops(pid, access, extra, write=True, cleanup=True, owner_words=3, protocol=ATOMIC))
        branch.add_op(scf.YieldOp())
        steps.add_ops([literal, selected, scf.IfOp(selected.result, [], Region(branch), Region())])
    steps.add_op(scf.YieldOp())
    outer.add_op(scf.ForOp(zero, upper.result, one, [], Region(steps)))
    observed = memref.LoadOp.build(operands=[accepted, [zero]], result_types=[builtin.i1])
    outer.add_ops([observed, memref.StoreOp.build(operands=[observed.res, results, [outer.args[0]]]), scf.YieldOp()])
    entry.add_op(scf.ForOp(zero, stop, one, [], Region(outer)))
    # Keep every actual emitted check and atomic state transition, but collect
    # assertion predicates so all histories can be measured in one compilation.
    for op in list(entry.walk()):
        if not isinstance(op, cf.AssertOp):
            continue
        previous = memref.LoadOp.build(operands=[accepted, [zero]], result_types=[builtin.i1])
        both = arith.AndIOp(previous.res, op.operands[0])
        stored = memref.StoreOp.build(operands=[both.result, accepted, [zero]])
        Rewriter.replace_op(op, [previous, both, stored])
    output = ToTensorOp.build(operands=[results], result_types=[builtin.TensorType(builtin.i1, [count])],
                              properties={"restrict": builtin.UnitAttr(), "writable": builtin.UnitAttr()})
    entry.add_ops([output, func.ReturnOp(output.tensor)])
    function = func.FuncOp("forward", ([a.type for a in entry.args], [output.tensor.type]), Region(entry))
    function.attributes["prov.memory_guard_scratch_bytes"] = builtin.IntegerAttr(64, builtin.i64)
    module = builtin.ModuleOp([function])
    module.verify()
    return module


@pytest.mark.skipif(not NATIVE, reason="existing native compiler environment required")
def test_native_sticky_guards_match_all_bounded_histories(tmp_path):
    from t2m.ir import text
    from t2m.native import execute_native
    cases = histories()
    pids, actions = np.zeros((len(cases), 4), np.int64), np.zeros((len(cases), 4), np.int64)
    for i, history in enumerate(cases):
        for j, (pid, action) in enumerate(history):
            pids[i, j], actions[i, j] = pid, action
    lengths = np.asarray([len(h) for h in cases], np.int64)
    expected = np.asarray([permitted(h) for h in cases], np.bool_)
    (actual,), _ = execute_native(text(batch_module(len(cases))), [pids, actions, lengths], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual, expected)
