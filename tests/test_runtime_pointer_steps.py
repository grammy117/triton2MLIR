"""Two source pointer streams and their final observations reach real consumers."""

import json
import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/native tools required")
CASES = ((0, 8, 1), (1, 8, 3), (3, 1, 2), (0, 0, 1), (2, 7, 2), (3, 8, 1))


def capture(root, transpose=False, dtype="i32", profile="tensor", policy="structured", *, bounds=None, rows=16, final_only=False, mix=True):
    from t2m.examples.runtime_header_kernels import bidirectional_sum, final_pointer_mix

    kernel = final_pointer_mix if final_only else bidirectional_sum
    constants = {"N": 17, "R": 16, "B": 8, "TRANSPOSE": transpose}
    if final_only:
        constants["MIX"] = mix
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel,
        {**dict.fromkeys(("x", "out", "last"), "*i64"), **dict.fromkeys(("lower", "upper", "stride"), dtype),
         **dict.fromkeys(constants, "constexpr")}, constants))
    bounds = ((0, 3), (0, 8), (1, 3)) if bounds is None else bounds
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", (17, rows) if transpose else (rows, 17), "read", "i64"),
        TensorBuffer("out", (17,), "write", "i64"), TensorBuffer("last", (17,), "write", "i64")), grid=(3,),
        runtime_scalars=tuple(RuntimeScalar(name, dtype, index, bounds=interval)
            for name, index, interval in zip(("lower", "upper", "stride"), (3, 4, 5), bounds, strict=True)))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile, address_policy=policy)


def arguments(transpose, dtype, index, *, final_only=False, mix=True):
    lower, upper, stride = CASES[index]
    x = np.arange(16 * 17, dtype=np.int64).reshape(16, 17) * (37 + index) + (1 << 55)
    x[1::2] *= -1
    x.flat[index], x.flat[-1] = -(1 << 63), (1 << 63)-1
    rows = np.asarray(list(range(lower, upper, stride)), dtype=np.int64)
    exit_row = lower + len(rows) * stride
    expected = [np.sum(x[rows] ^ (x[15-rows] * np.int64(3)), axis=0, dtype=np.int64),
                x[exit_row] - x[15-exit_row] * np.int64(3)]
    if final_only:
        state = x[0].copy()
        if mix:
            for row in rows:
                state = (state ^ np.int64(row + 2)) * np.int64(3)
        expected[0] = state ^ x[exit_row]
    integer = np.int32 if dtype == "i32" else np.int64
    return [x.T.copy() if transpose else x, *[np.asarray(value, integer) for value in (lower, upper, stride)]], expected


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype", ["i32", "i64"])
@pytest.mark.parametrize("profile,policy", [("tensor", "structured"), ("buffer", "structured"),
    ("guarded-buffer", "structured"), ("guarded-buffer", "source-words")])
def test_streams_final_pointers_and_original_numeric_words_execute(tmp_path, transpose, dtype, profile, policy):
    from t2m.native import execute_native

    result = capture(tmp_path, transpose, dtype, profile, policy)
    if policy == "structured":
        report = next(row for row in result.trace["transformations"] if row["kind"] == "source-step-coupled-pointer-recurrences")
        assert [row["step_coefficient"] for row in report["streams"]] == ([1, -1] if transpose else [17, -17])
    if profile == "tensor":
        assert not any(op.name.startswith("memref.") for op in result.module.walk())
        assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
        assert result.trace["memory"]["source_loops"][0]["actual_exit_induction_results"]
    for index in (1, 2, 3):
        inputs, expected = arguments(transpose, dtype, index)
        actual, _ = execute_native(result.mlir_text, inputs, tmp_path / f"native-{index}",
            python=os.environ["T2M_NATIVE_PYTHON"])
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())


@pytest.mark.parametrize("bounds,rows", [(((0, 3), (0, 8), (0, 3)), 16),
    (((0, 3), (0, 8), (1, 1 << 30)), 16), (((0, 3), (0, 8), (1, 3)), 10),
    (((0, 3), (0, 16), (1, 3)), 16)])
def test_invalid_source_steps_overflow_and_final_pointer_bounds_still_refuse(tmp_path, bounds, rows):
    with pytest.raises(ConversionError):
        capture(tmp_path, bounds=bounds, rows=rows)


@pytest.fixture(scope="module", params=[(transpose, dtype) for transpose in (False, True) for dtype in ("i32", "i64")])
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing Merlin tools required")
    transpose, dtype = request.param
    root = tmp_path_factory.mktemp(f"runtime-pointer-{transpose}-{dtype}")
    result = capture(root, transpose, dtype)
    inputs, expected = arguments(transpose, dtype, 1)
    write_shared_bundle(result, root / "capture", inputs, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    compatibility = json.loads((root / "capture/compatibility.json").read_text())
    assert any("source pointer streams proved" in row for row in compatibility["differences"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, transpose, dtype


def test_one_merlin_library_preserves_runtime_strides_and_every_final_observation(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    root, transpose, dtype = compiled
    library = _record(root / "merlin/host/model.so")
    for index in range(len(CASES)):
        inputs, expected = arguments(transpose, dtype, index)
        arrays, reference = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
        np.savez(arrays, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        report = invoke_kernel(root / "merlin", arrays, reference, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(root / f"invoke-{index}")
    assert _record(root / "merlin/host/model.so") == library


def test_actual_serial_images_keep_nonunit_strides_and_zero_trip_final_pointers(compiled):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial tools required")
    root, transpose, dtype = compiled
    for index in (1, 2):
        inputs, expected = arguments(transpose, dtype, index)
        arrays, reference = root / f"target-arguments-{index}.npz", root / f"target-reference-{index}.npz"
        np.savez(arrays, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=arrays, reference=reference, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
