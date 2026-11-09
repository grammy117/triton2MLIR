"""Exercise real guard IR at 31/62-bit program boundaries without huge grids."""

import os

import numpy as np
import pytest

from t2m.memory_contract import guard_layout

NATIVE = os.environ.get("T2M_NATIVE_PYTHON")


@pytest.mark.parametrize("grid,words,cost", [((1,), 1, 16), ((2**31 - 1,), 1, 16),
    ((65536, 65536), 2, 24), ((2**31 - 1, 2**31 - 1, 2), 3, 32)])
def test_owner_storage_spans_the_complete_declared_grid(grid, words, cost):
    assert guard_layout({"grid": grid}) == {"owner_words_per_element": words, "guard_storage_bytes_per_output_element": cost}


@pytest.mark.skipif(not NATIVE, reason="existing native compiler environment required")
@pytest.mark.parametrize("programs,writes,accept", [
    ((0, 1 << 31), (False, False), True), ((0, 1 << 62), (False, False), True),
    ((1 << 62, 1 << 62), (False, True), True), ((1 << 31, 1 << 31), (True, False), True),
    ((0, 1 << 31), (False, True), False), ((1 << 31, 0), (True, False), False),
    ((0, 1 << 62), (False, True), False), ((1 << 62, 0), (True, False), False)])
def test_native_ownership_checks_all_program_fragments(tmp_path, programs, writes, accept):
    from xdsl.dialects import arith, builtin, func, memref, tensor
    from xdsl.dialects.linalg.ops import FillOp
    from xdsl.ir import Block, Region
    from t2m.guarded_memory import Access, guard_loops
    from t2m.ir import text
    from t2m.native import execute_native
    from t2m.shared_tensor import TensorBuffer
    entry = Block(arg_types=[builtin.i64, builtin.i64])
    def constant(value, type_=builtin.i64):
        op = arith.ConstantOp.from_int_and_width(value, type_)
        entry.add_op(op)
        return op.result
    zero, one = constant(0), constant(1)
    storage = []
    for size in (3, 1):
        op = memref.AllocOp.build(operands=[[], []], result_types=[builtin.MemRefType(builtin.i64, [size])])
        entry.add_ops([op, FillOp([zero], [op.memref], res=[])])
        storage.append(op.memref)
    access = Access(TensorBuffer("out", (1,), "readwrite"), (1,), offsets=(0,), strides=(1,))
    source = arith.ConstantOp.from_int_and_width(0, builtin.i64)
    for pid, write in zip(entry.args, writes, strict=True):
        extra = {("extent", "out"): one, ("owners", "out"): storage[0], ("visited", "out"): storage[1], ("program", None): pid}
        entry.add_ops(guard_loops(source, access, extra, write=write, owner_words=3))
        if write:
            entry.add_ops(guard_loops(source, access, extra, write=True, cleanup=True, owner_words=3))
    result = tensor.FromElementsOp(zero)
    entry.add_ops([result, func.ReturnOp(result.result)])
    function = func.FuncOp("forward", ([builtin.i64, builtin.i64], [builtin.TensorType(builtin.i64, [1])]), Region(entry))
    function.attributes["prov.memory_guard_scratch_bytes"] = builtin.IntegerAttr(32, builtin.i64)
    module = builtin.ModuleOp([function])
    module.verify()
    values = [np.asarray(pid, dtype=np.int64) for pid in programs]
    if accept:
        (actual,), _ = execute_native(text(module), values, tmp_path / "native", python=NATIVE)
        np.testing.assert_array_equal(actual, np.array([0], dtype=np.int64))
    else:
        with pytest.raises(ValueError, match=r"native host execution failed \(exit -6\)"):
            execute_native(text(module), values, tmp_path / "native", python=NATIVE)
