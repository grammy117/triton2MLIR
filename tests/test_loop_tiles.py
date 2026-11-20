"""Actual source loops retain partial tile state, observations and zero trips."""

import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/native tools required")
MODES = ("positive", "zero-before", "zero-after", "complete", "nested")


def capture(root, mode):
    from t2m.examples.loop_tile_kernels import loop_partition

    source = root / "source.ttir"
    source.write_text(capture_ttir(loop_partition,
        {**{name: "*i64" for name in ("x", "bias", "out", "seen")}, "n": "i32", "steps": "i32",
         "B": "constexpr", "MODE": "constexpr"}, {"B": 8, "MODE": mode}))
    abi = TensorABI("loop_partition", (TensorBuffer("x", ("n",), "read", "i64"),
        TensorBuffer("bias", (1,), "read", "i64"), TensorBuffer("out", ("n",), "write", "i64"),
        TensorBuffer("seen", ("n",), "write", "i64")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
                         RuntimeScalar("steps", "i32", 5, bounds=(-2 if mode in (
                             "zero-before", "zero-after", "complete", "missing-zero") else 1, 4))))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def reference(x, bias, steps, mode):
    word = x ^ bias[0]
    positive = word > 0
    state, snapshot = word ^ 1, np.zeros_like(word)
    for iteration in range(steps):
        earlier = np.where(positive, state, -17)
        update = (word ^ (iteration + (32 if mode == "nested" and iteration % 2 else 2))) ^ earlier
        state = np.where(positive, state, update)
        snapshot ^= state
        state = np.where(positive, state ^ 8, state)
    if steps <= 0 and mode in ("zero-before", "zero-after"):
        state = np.where(positive, state, word ^ 16)
    return [state, snapshot]


def arguments(mode, index):
    n = (0, 1, 8, 17, 31, 9)[index]
    steps = (0, -2, 1, 3, 0, 2)[index] if mode in ("zero-before", "zero-after", "complete") else (1, 4, 1, 3, 4, 2)[index]
    x = np.arange(n, dtype=np.int64) * 37 + (1 << 56)
    x[1::2] *= -1
    if n:
        x[-1] = (0, (1 << 63)-1, -(1 << 63), -1, 0, -65535)[index]
    bias = np.asarray([(1 << 55) + index], np.int64)
    return [x, bias, np.asarray(n, np.int32), np.asarray(steps, np.int32)], reference(x, bias, steps, mode)


@pytest.mark.parametrize("mode", MODES)
def test_source_loops_keep_tensor_state_and_complete_original_correspondence(tmp_path, mode):
    result = capture(tmp_path, mode)
    assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    reports = [row for row in result.trace["transformations"] if row["kind"] == "source-loop-tile-state"]
    assert reports and all(row["root"] == "out" and row["observed_read_ids"] and row["loop_source_ids"] for row in reports)
    loops = [op for op in result.module.walk() if op.name == "scf.for"]
    assert loops and any(len(op.results) >= 2 for op in loops)
    assert "source-loop-tile-state" in str(result.trace)


@pytest.mark.parametrize("mode", ("missing-zero", "early-body", "wrong-index", "incomplete-body", "repeat-program"))
def test_zero_path_premature_reads_addresses_and_parallel_conflicts_still_refuse(tmp_path, mode):
    with pytest.raises(ConversionError):
        capture(tmp_path, mode)


@pytest.mark.parametrize("mode", MODES)
def test_native_loop_recurrence_observations_and_empty_domains(tmp_path, mode):
    from t2m.native import execute_native

    result = capture(tmp_path, mode)
    for index in range(6):
        inputs, expected = arguments(mode, index)
        actual, _report = execute_native(result.mlir_text, inputs, tmp_path / f"native-{index}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())


@pytest.fixture(scope="module", params=MODES)
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing Merlin tools required")
    root = tmp_path_factory.mktemp("loop-tile-" + request.param)
    result = capture(root, request.param)
    inputs, expected = arguments(request.param, 3)
    write_shared_bundle(result, root / "capture", inputs, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, request.param


def test_one_merlin_library_preserves_every_iteration_and_zero_trip_source_state(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import verify_kernel_qualification

    root, mode = compiled
    library, receipt = _record(root / "merlin/host/model.so"), verify_kernel_qualification(root / "merlin")
    for index in range(6):
        inputs, expected = arguments(mode, index)
        arguments_file, reference_file = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
        np.savez(arguments_file, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference_file, **{f"out{i}": value for i, value in enumerate(expected)})
        report = invoke_kernel(root / "merlin", arguments_file, reference_file, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(root / f"invoke-{index}")
    assert _record(root / "merlin/host/model.so") == library and verify_kernel_qualification(root / "merlin") == receipt


def test_actual_serial_images_preserve_loop_observations_and_source_zero_paths(compiled):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    root, mode = compiled
    for index in (4, 5):
        inputs, expected = arguments(mode, index)
        arguments_file, reference_file = root / f"target-arguments-{index}.npz", root / f"target-reference-{index}.npz"
        np.savez(arguments_file, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference_file, **{f"out{i}": value for i, value in enumerate(expected)})
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=arguments_file, reference=reference_file, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
