"""Serial scalar source events retain ordered tensor versions and fresh results."""

import json
import os

import numpy as np
import pytest
from test_mutating_view_models import implementation

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = bool(os.environ.get("TRITON_SHARED_OPT") and os.environ.get("T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing shared/native tools required")


@pytest.mark.skipif(not os.environ.get("TRITON_SHARED_OPT"), reason="existing shared optimizer required")
def test_real_mutable_views_have_a_standard_tensor_payload(tmp_path):
    kernel = implementation(tmp_path / "kernel", "matrix", profile="tensor").implementation
    names = {op.name for op in kernel.module.walk()}
    assert {"tensor.insert", "tensor.extract", "scf.for"} <= names
    assert not any(name.startswith("memref.") for name in names)
    assert kernel.trace["memory"]["output_initialization"]["old_left"]["coverage"] == "serial-contiguous-source-loop"


def test_all_unit_ranked_roots_retain_their_existing_tensor_axes(tmp_path):
    from t2m.examples.serial_memory_kernels import single_ranked_tile
    from t2m.native import execute_native

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(single_ranked_tile, {"x": "*i64", "out": "*i64"}, {}))
    abi = TensorABI("single_ranked_tile", (TensorBuffer("x", (1, 1), "read", "i64"), TensorBuffer("out", (1, 1), "write", "i64")))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    initial = np.asarray([[1 << 56]], dtype=np.int64)
    actual, _ = execute_native(result.mlir_text, [initial], tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    assert len(actual) == 1
    assert (actual[0].shape, actual[0].dtype, actual[0].tobytes()) == (initial.shape, initial.dtype, (initial + 3).tobytes())


def convert(root, *, mode="full", dtype="i64", grid=(1,), profile="tensor"):
    from t2m.examples.serial_memory_kernels import serial_snapshot

    root.mkdir()
    source = root / "source.ttir"
    source.write_text(capture_ttir(serial_snapshot, {**{name: "*" + ("fp64" if dtype == "f64" else dtype) for name in ("state", "before", "after", "seen")},
        **{name: "i32" for name in ("n", "count", "position", "enabled")}, "MODE": "constexpr"}, {"MODE": mode}))
    scalars = [RuntimeScalar("n", "i32", 4, bounds=(0, 31))]
    scalars.append(RuntimeScalar("count", "i32", 5, bounds=(0, 31)))
    scalars.extend((RuntimeScalar("position", "i32", 6, bounds=(-31, 31)), RuntimeScalar("enabled", "i32", 7, bounds=(0, 1))))
    abi = TensorABI("serial_snapshot", (TensorBuffer("state", ("n",), "readwrite", dtype),
        TensorBuffer("before", ("n",), "write", dtype), TensorBuffer("after", ("n",), "write", dtype),
        TensorBuffer("seen", (2,), "write", dtype)), runtime_scalars=tuple(scalars), grid=grid)
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile=profile)


@pytest.mark.parametrize("mode,count", (("full", "different"), ("masked", "same"), ("branch", "same"),
                                      ("duplicate", "same"), ("before-read", "same")))
def test_incomplete_or_unobserved_fresh_storage_does_not_get_loop_credit(tmp_path, mode, count):
    with pytest.raises(ConversionError):
        convert(tmp_path / "case", mode="different" if count == "different" else mode)


def test_parallel_scalar_writes_require_their_original_ownership_proof(tmp_path):
    with pytest.raises(ConversionError):
        convert(tmp_path / "case", grid=(2,))


def test_unguarded_buffer_continuation_cannot_inherit_tensor_runtime_checks(tmp_path):
    with pytest.raises(ConversionError, match="(gather/scatter|all-unit)"):
        convert(tmp_path / "case", profile="buffer")


@pytest.mark.parametrize("mode,position", (("full", -1), ("full", 7), ("bad-store", 0)))
def test_active_scalar_bounds_abort_before_extraction_or_insertion(tmp_path, mode, position):
    from t2m.native import execute_native

    result = convert(tmp_path / "case", mode=mode)
    arrays = [np.arange(7, dtype=np.int64), *(np.asarray(value, np.int32) for value in (7, 7, position, 1))]
    with pytest.raises(ValueError, match="native host execution failed") as failure:
        execute_native(result.mlir_text, arrays, tmp_path / "failed", python=os.environ["T2M_NATIVE_PYTHON"],
                       shape_abi=result.spec.runtime_shapes())
    message = str(failure.value)
    assert "exit -6" in message
    assert ("Triton scalar scatter active index" if mode == "bad-store" else "Triton gather active index") in message
    assert not (tmp_path / "failed/outputs.npz").exists()
    (tmp_path / "abort.json").write_text(json.dumps({"mode": mode, "position": position, "error": message}, indent=2) + "\n")


@pytest.mark.parametrize("dtype", ("i64", "f64"))
def test_serial_updates_keep_snapshots_empty_masks_and_source_words(tmp_path, dtype):
    from t2m.native import execute_native

    result = convert(tmp_path / "case", dtype=dtype)
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    for index, (n, position, enabled) in enumerate(((0, -31, 0), (1, 0, 1), (7, 3, 1), (31, 30, 1), (13, -31, 0), (7, -31, 0))):
        initial = np.arange(n, dtype=np.int64 if dtype == "i64" else np.float64) + ((1 << 55) if dtype == "i64" else 0.25)
        if dtype == "f64" and index == 5:
            initial = np.asarray([0, 1 << 63, 1, 0x7ff0000000000000, 0xfff0000000000000,
                                  0x7ff0000000000001, 0xfff8123456789abc], dtype=np.uint64).view(np.float64)
        current = initial.copy()
        if enabled:
            old = current[position].copy()
            current[position] += 7
            new = current[position].copy()
        else:
            old, new = -19, -23
        expected = [current, initial, initial, np.asarray([old, new], dtype=initial.dtype)]
        actual, _ = execute_native(result.mlir_text, [initial, np.asarray(n, np.int32), np.asarray(n, np.int32), np.asarray(position, np.int32),
            np.asarray(enabled, np.int32)], tmp_path / f"native-{index}", python=os.environ["T2M_NATIVE_PYTHON"],
            shape_abi=result.spec.runtime_shapes())
        assert len(actual) == len(expected)
        for observed, wanted in zip(actual, expected, strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
