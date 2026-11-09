"""Loaded-index reads retain exact masks/fills and guard active logical bounds."""

import json
import os

import numpy as np
import pytest
from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import DenseArrayBase

from t2m.bundle import verify_bundle
from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.ir import text
from t2m.native import execute_native
from t2m.shared_bundle import write_shared_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared, opcode, parse_shared
from t2m.upstream import lower_with_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing native/shared environments required")


def capture(root, *, dtype="i32", masked=True, shape=(3, 5), effect="read"):
    from t2m.examples.tensor_kernels import indexed_gather
    path = root / "kernel.ttir"
    path.write_text(capture_ttir(indexed_gather,
        {"x": "*fp32", "indices": "*" + dtype, "out": "*fp32", "N": "constexpr", "B": "constexpr", "MASKED": "constexpr"},
        {"N": 13, "B": 8, "MASKED": masked}))
    abi = TensorABI("indexed_gather", (TensorBuffer("x", shape, effect), TensorBuffer("indices", (13,), dtype=dtype),
                    TensorBuffer("out", (13,), "write")), grid=(2,))
    return path, abi


def indices(dtype, masked):
    values = np.array([0, 14, 7, 2, 3, 3, 11, 8, 0, 14, 12, 5, 6], dtype=dtype)
    if masked:
        values[[3, 6, 11]] = [np.iinfo(dtype).min, -99, -1]
    return values


def reference(x, index):
    answer = np.full(index.shape, -7, dtype=np.float32)
    active = index >= 0
    answer[active] = x.ravel()[index[active]]
    return answer


@pytest.mark.parametrize("dtype,shape,masked", [("i32", (3, 5), True), ("i64", (15, 1), True),
    ("i32", (15,), False), ("i64", (1, 3, 5, 1), True)])
def test_irregular_reads_masks_tails_duplicates_and_unit_dimensions(tmp_path, dtype, shape, masked):
    path, abi = capture(tmp_path, dtype=dtype, masked=masked, shape=shape)
    result = convert_shared(path, abi, tmp_path / "translation", optimizer=OPT)
    x = np.arange(15, dtype=np.float32).reshape(shape)
    index = indices(np.int32 if dtype == "i32" else np.int64, masked)
    write_shared_bundle(result, tmp_path / "bundle", [x, index], [reference(x, index)], work=tmp_path / "native", python=NATIVE)
    verify_bundle(tmp_path / "bundle")
    facts = result.trace["memory"]["source_gathers"]
    assert facts and facts[0]["bounds"] == "runtime-assert-before-active-extract"
    assert facts[0]["source_ids"]
    assert {"scf.forall", "linalg.generic", "scf.if", "tensor.extract", "cf.assert"} <= {op.name for op in result.module.walk()}
    guards = [op for op in result.module.walk() if op.name == "cf.assert"]
    assert all(op.attributes["prov.family"].data == "memory" for op in guards)
    assert all(op.parent_op().name == "scf.if" for op in guards)
    # Runtime indices are data, so a second all-inactive input uses the exact
    # same emitted program and never extracts an out-of-bounds logical element.
    if masked:
        inactive = np.full(13, np.iinfo(index.dtype).min, dtype=index.dtype)
        (actual,), _ = execute_native(result.mlir_text, [x, inactive], tmp_path / "inactive", python=NATIVE)
        np.testing.assert_array_equal(actual, np.full(13, -7, dtype=np.float32))


@pytest.mark.parametrize("masked,bad", [(False, -1), (True, 15)])
def test_active_out_of_bounds_index_fails_before_extract(tmp_path, masked, bad):
    path, abi = capture(tmp_path, masked=masked)
    result = convert_shared(path, abi, tmp_path / "translation", optimizer=OPT)
    index = indices(np.int32, masked)
    index[0] = bad
    with pytest.raises(ValueError, match="native host execution failed"):
        execute_native(result.mlir_text, [np.arange(15, dtype=np.float32).reshape(3, 5), index], tmp_path / "native", python=NATIVE)


def test_unmasked_indirect_load_without_fill_retains_checked_extraction(tmp_path):
    path, abi = capture(tmp_path, masked=False)
    lower_with_shared(path, tmp_path / "generic", optimizer=OPT, stage="source")
    module = parse_shared((tmp_path / "generic/lowered.mlir").read_text())
    load = [op for op in module.walk() if opcode(op) == "tt.load"][-1]
    load.operands = (load.operands[0],)
    load.properties["operandSegmentSizes"] = DenseArrayBase.from_list(builtin.i32, [1, 0, 0])
    path.write_text(text(module))
    result = convert_shared(path, abi, tmp_path / "translation", optimizer=OPT)
    x, index = np.arange(15, dtype=np.float32).reshape(3, 5), indices(np.int32, False)
    (actual,), _ = execute_native(result.mlir_text, [x, index], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual, reference(x, index))


@pytest.mark.parametrize("change", ["undefined_fill", "scatter", "overflow", "writable_read"])
def test_unproved_irregular_memory_is_not_silently_functionalized(tmp_path, change):
    path, abi = capture(tmp_path, effect="readwrite" if change == "writable_read" else "read")
    lower_with_shared(path, tmp_path / "generic", optimizer=OPT, stage="source")
    module = parse_shared((tmp_path / "generic/lowered.mlir").read_text())
    loads = [op for op in module.walk() if opcode(op) == "tt.load"]
    gather, indexed_pointer = loads[-1], loads[-1].operands[0].owner
    if change == "undefined_fill":
        gather.operands = tuple(gather.operands[:2])
        gather.properties["operandSegmentSizes"] = DenseArrayBase.from_list(builtin.i32, [1, 1, 0])
    elif change == "scatter":
        store = next(op for op in module.walk() if opcode(op) == "tt.store")
        pointer = store.operands[0].owner
        pointer.operands = (pointer.operands[0], indexed_pointer.operands[1])
    elif change == "overflow":
        value = indexed_pointer.operands[1]
        constant = arith.ConstantOp(builtin.DenseIntOrFPElementsAttr.from_list(value.type, [2]))
        product = arith.MuliOp(value, constant)
        indexed_pointer.parent.insert_ops_before([constant, product], indexed_pointer)
        indexed_pointer.operands = (indexed_pointer.operands[0], product.result)
    path.write_text(text(module))
    reason = {"undefined_fill": "explicit defined fill", "scatter": "(qualified structured descriptor|gather/scatter)",
              "overflow": "may overflow", "writable_read": "gather/scatter"}[change]
    with pytest.raises(ConversionError, match=reason):
        convert_shared(path, abi, tmp_path / "translation", optimizer=OPT)


@pytest.mark.skipif(not all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")),
                    reason="existing Merlin native toolchain required")
def test_indexed_gather_has_working_merlin_consumer_with_unchanged_reference(tmp_path):
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    path, abi = capture(tmp_path, dtype="i64", shape=(15, 1))
    result = convert_shared(path, abi, tmp_path / "translation", optimizer=OPT)
    x, index = np.arange(15, dtype=np.float32).reshape(15, 1), indices(np.int64, True)
    bundle, out = tmp_path / "bundle", tmp_path / "merlin"
    write_shared_bundle(result, bundle, [x, index], [reference(x, index)], work=tmp_path / "native", python=NATIVE)
    qualify_kernel(bundle, out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
                   compiler_python=NATIVE, clang=os.environ["T2M_MERLIN_CLANG"])
    verify_kernel_qualification(out)
    report = json.loads((out / "host/execution.json").read_text())
    assert report["evaluation"]["functional_inputs_preserved"] is True
