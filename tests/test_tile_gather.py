"""Two-axis value gathers, exact transport and invalid-index checks through Merlin."""

import os

import numpy as np
import pytest

from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
PROFILES = ["tensor", "buffer", "guarded-buffer"]


def capture(root, profile, axis, dtype="f32", index_dtype="i32"):
    from t2m.examples.tile_gather_kernels import tile_gather_values
    from t2m.frontend import capture_ttir
    shape = (2, 8) if axis == 0 else (4, 4)
    constants = {"AXIS": axis, "R": 4, "S": 8, "OR": shape[0], "OS": shape[1]}
    source = root / "source.ttir"
    pointer = "*fp32" if dtype == "f32" else "*i32"
    source.write_text(capture_ttir(tile_gather_values, {"x": pointer, "indices": "*" + index_dtype, "out": pointer,
        **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("tile_gather_values", (TensorBuffer("x", (4, 8), dtype=dtype),
        TensorBuffer("indices", shape, dtype=index_dtype), TensorBuffer("out", shape, "write", dtype=dtype)))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile=profile,
                          address_policy="source-words" if profile == "guarded-buffer" else "structured")


def arguments(axis, case, dtype="f32", index_dtype="i32"):
    shape = (2, 8) if axis == 0 else (4, 4)
    if dtype == "f32":
        bits = np.array([0x80000000, 0, 0x7f800000, 0xff800000, 0x7fc12345, 0x3f800000], dtype=np.uint32)
        x = np.resize(np.roll(bits, case), 32).view(np.float32).reshape(4, 8).copy()
    else:
        x = np.arange(32, dtype=np.int32).reshape(4, 8) * 17 + case
    indices = ((np.indices(shape).sum(axis=0) * 3 + case) % x.shape[axis]).astype("int64" if index_dtype == "i64" else "int32")
    return [x, indices], np.take_along_axis(x, indices, axis)


def assert_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype,index_dtype", [("f32", "i32"), ("i32", "i32"), ("f32", "i64")])
def test_tile_gather_keeps_repeated_indices_and_exact_values(tmp_path, profile, axis, dtype, index_dtype):
    from t2m.native import execute_native
    result = capture(tmp_path, profile, axis, dtype, index_dtype)
    assert any(r["opcode"] == "tt.gather" for r in result.trace["source_operations"])
    assert any(t["kind"] == "ttir-tile-gather" and t["axis"] == axis for t in result.trace["transformations"])
    for case in range(3):
        arrays, expected = arguments(axis, case, dtype, index_dtype)
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{case}", python=NATIVE)
        assert_bits(actual, expected)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("index_dtype", ["i32", "i64"])
def test_tile_gather_checks_indices_before_extraction(tmp_path, profile, axis, index_dtype):
    from t2m.native import execute_native
    result = capture(tmp_path, profile, axis, index_dtype=index_dtype)
    invalids = (-1, (4, 8)[axis], 2**32 + 1, -(2**32) + 1) if index_dtype == "i64" else (-1, (4, 8)[axis])
    for i, invalid in enumerate(invalids):
        arrays, _ = arguments(axis, 0, index_dtype=index_dtype)
        arrays[1][0, 0] = invalid
        with pytest.raises(ValueError, match="exit -6"):
            execute_native(result.mlir_text, arrays, tmp_path / f"invalid{i}", python=NATIVE)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("index_dtype", ["i32", "i64"])
def test_one_merlin_library_retains_tile_gather_and_value_bits(tmp_path, profile, axis, index_dtype):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result = capture(tmp_path, profile, axis, index_dtype=index_dtype)
    arrays, expected = arguments(axis, 0, index_dtype=index_dtype)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for case in range(3):
        arrays, expected = arguments(axis, case, index_dtype=index_dtype)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        output = tmp_path / f"invoke{case}"
        report = invoke_kernel(compiled, inputs, reference, output, merlin_root=root, python=python)
        assert report["evaluation"]["functional_inputs_preserved"]
        with np.load(output / "evaluation/outputs.npz", allow_pickle=False) as data:
            assert_bits(data["out0"], expected)
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library
