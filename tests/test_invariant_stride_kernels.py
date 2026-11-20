"""Runtime memory progress remains separate from actual loop control."""

import json
import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(k) for k in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/native tools required")
MODES = ("memory", "pure", "mixed", "late")
CASES = ((0, 8, 1, -2), (1, 8, 3, 2), (3, 1, 2, -2), (0, 0, 1, 0), (2, 7, 2, -1), (3, 8, 1, 0), (0, 1, 1, 1))


def capture(root, mode="memory", transpose=False, dtype="i32", profile="tensor", policy="structured", *, memory_bounds=(-2, 2)):
    from t2m.examples.runtime_stride_kernels import invariant_stride

    constants = {"N": 17, "R": 64, "B": 8, "TRANSPOSE": transpose, "MODE": mode}
    source = root / "source.ttir"
    source.write_text(capture_ttir(invariant_stride,
        {**dict.fromkeys(("x", "out", "last"), "*i64"),
         **dict.fromkeys(("lower", "upper", "stride", "memory_stride"), dtype),
         **dict.fromkeys(constants, "constexpr")}, constants))
    abi = TensorABI("invariant_stride", (TensorBuffer("x", (17, 64) if transpose else (64, 17), "read", "i64"),
        TensorBuffer("out", (17,), "write", "i64"), TensorBuffer("last", (17,), "write", "i64")), grid=(3,),
        runtime_scalars=tuple(RuntimeScalar(name, dtype, i, bounds=bounds) for i, (name, bounds) in enumerate(
            zip(("lower", "upper", "stride", "memory_stride"), ((0, 3), (0, 8), (1, 3), memory_bounds), strict=True), 3)))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile, address_policy=policy)


def reference(x, controls, mode):
    lo, hi, step, memory_stride = map(int, controls)
    rows = list(range(lo, hi, step))
    increment = step + memory_stride if mode == "mixed" else (1 // hi if rows else 0) if mode == "late" else memory_stride
    position, state = 16 + lo, x[0].copy()
    for row in rows:
        if mode == "pure":
            state = (state ^ np.int64(row + 2)) * np.int64(3)
        else:
            state = state + (x[position] ^ (x[63-position] * np.int64(3)))
        position += increment
    return state ^ x[position], x[position] - x[63-position] * np.int64(3)


def arguments(mode, transpose, dtype, index):
    controls = CASES[index]
    x = np.arange(64*17, dtype=np.int64).reshape(64, 17) * (37 + index) + (1 << 55)
    x[1::2] *= -1
    x.flat[16*17], x.flat[47*17+1] = -(1 << 63), (1 << 63)-1
    expected = reference(x, controls, mode)
    integer = np.int32 if dtype == "i32" else np.int64
    return [x.T.copy() if transpose else x, *[np.asarray(v, integer) for v in controls]], expected


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("transpose,dtype", [(False, "i32"), (True, "i64")])
@pytest.mark.parametrize("profile,policy", [("tensor", "structured"), ("buffer", "structured"),
    ("guarded-buffer", "structured"), ("guarded-buffer", "source-words")])
def test_signed_memory_progress_original_controls_and_final_reads(tmp_path, mode, transpose, dtype, profile, policy):
    from t2m.native import execute_native

    result = capture(tmp_path, mode, transpose, dtype, profile, policy)
    if policy == "structured":
        proof = next(row for row in result.trace["transformations"] if row["kind"] == "source-invariant-runtime-pointer-recurrences")
        assert [row["logical_unit_factor"] for row in proof["streams"]] == [1 if transpose else 17] * 2
        assert "no-hoisting" in proof["evaluation"]
    if profile == "tensor":
        assert not any(op.name.startswith(("memref.", "gpu.")) for op in result.module.walk())
        assert "scf.forall" in result.mlir_text
        assert any(row.get("actual_invariant_recurrence_results") for row in result.trace["memory"]["source_loops"])
    for index in ((0, 1, 3, 6) if mode == "late" else (0, 1, 2, 5)):
        arrays, expected = arguments(mode, transpose, dtype, index)
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native-{index}", python=os.environ["T2M_NATIVE_PYTHON"])
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())


@pytest.mark.parametrize("mode", ("memory", "pure", "mixed"))
def test_final_memory_coordinates_still_need_independent_root_bounds(tmp_path, mode):
    with pytest.raises(ConversionError):
        capture(tmp_path, mode, memory_bounds=(-8, 8))


@pytest.fixture(scope="module", params=[(m, t, d) for m in MODES for t, d in ((False, "i32"), (True, "i64"))])
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(k) for k in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing Merlin tools required")
    mode, transpose, dtype = request.param
    root = tmp_path_factory.mktemp(f"invariant-stride-{mode}-{transpose}-{dtype}")
    result = capture(root, mode, transpose, dtype)
    arrays, expected = arguments(mode, transpose, dtype, 1)
    write_shared_bundle(result, root / "capture", arrays, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    differences = json.loads((root / "capture/compatibility.json").read_text())["differences"]
    assert any("independent invariant runtime pointer increments" in row for row in differences)
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, mode, transpose, dtype


def test_unchanged_merlin_library_replays_independent_control_and_memory_strides(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    root, mode, transpose, dtype = compiled
    library = _record(root / "merlin/host/model.so")
    for index in range(len(CASES)):
        arrays, expected = arguments(mode, transpose, dtype, index)
        inputs, reference_path = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference_path, **{f"out{i}": value for i, value in enumerate(expected)})
        report = invoke_kernel(root / "merlin", inputs, reference_path, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(root / f"invoke-{index}")
    assert _record(root / "merlin/host/model.so") == library


def test_actual_serial_images_preserve_signed_progress_and_legal_zero_trips(compiled):
    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial tools required")
    root, mode, transpose, dtype = compiled
    for index in (0, 3):
        arrays, expected = arguments(mode, transpose, dtype, index)
        inputs, reference_path = root / f"target-arguments-{index}.npz", root / f"target-reference-{index}.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference_path, **{f"out{i}": value for i, value in enumerate(expected)})
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=inputs, reference=reference_path, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
