"""Pure numeric loops keep tile reuse, word state and final pointer observations."""

import json
import os

import numpy as np
import pytest
from test_runtime_pointer_steps import TOOLS, arguments, capture

from t2m.diagnostics import ConversionError

pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/native tools required")


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype", ["i32", "i64"])
@pytest.mark.parametrize("profile,policy", [("tensor", "structured"), ("buffer", "structured"),
    ("guarded-buffer", "structured"), ("guarded-buffer", "source-words")])
def test_original_pure_word_recurrence_and_final_reads_execute(tmp_path, transpose, dtype, profile, policy):
    from t2m.native import execute_native

    result = capture(tmp_path, transpose, dtype, profile, policy, final_only=True)
    if profile == "tensor":
        loop = result.trace["memory"]["source_loops"][0]
        assert loop["memory_effects"] == "no-buffer-accesses" and loop["actual_exit_induction_results"]
        assert not any(op.name.startswith("memref.") for op in result.module.walk())
        assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
        numeric_loop = next(op for op in result.module.walk() if op.name == "scf.for")
        assert not any(op.name == "tensor.extract_slice" for op in numeric_loop.walk())
    for index in (1, 2, 3):
        inputs, expected = arguments(transpose, dtype, index, final_only=True)
        actual, _ = execute_native(result.mlir_text, inputs, tmp_path / f"native-{index}", python=os.environ["T2M_NATIVE_PYTHON"])
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype", ["i32", "i64"])
def test_pointer_only_loop_keeps_preloaded_tile_without_an_extra_numeric_carrier(tmp_path, transpose, dtype):
    from t2m.native import execute_native

    result = capture(tmp_path, transpose, dtype, final_only=True, mix=False)
    for index in (1, 2):
        inputs, expected = arguments(transpose, dtype, index, final_only=True, mix=False)
        actual, _ = execute_native(result.mlir_text, inputs, tmp_path / f"native-{index}", python=os.environ["T2M_NATIVE_PYTHON"])
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())


def test_pure_cloning_cannot_justify_out_of_root_final_reads(tmp_path):
    with pytest.raises(ConversionError):
        capture(tmp_path, final_only=True, bounds=((0, 3), (0, 16), (1, 3)))


@pytest.fixture(scope="module", params=[(transpose, dtype) for transpose in (False, True) for dtype in ("i32", "i64")])
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing Merlin tools required")
    transpose, dtype = request.param
    root = tmp_path_factory.mktemp(f"pure-pointer-{transpose}-{dtype}")
    result = capture(root, transpose, dtype, final_only=True)
    inputs, expected = arguments(transpose, dtype, 1, final_only=True)
    write_shared_bundle(result, root / "capture", inputs, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    compatibility = json.loads((root / "capture/compatibility.json").read_text())
    assert any("pure numeric source loops" in row for row in compatibility["differences"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, transpose, dtype


def test_one_library_retains_pure_loop_state_and_every_final_pointer_read(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    root, transpose, dtype = compiled
    library = _record(root / "merlin/host/model.so")
    for index in range(6):
        inputs, expected = arguments(transpose, dtype, index, final_only=True)
        arrays, reference = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
        np.savez(arrays, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        report = invoke_kernel(root / "merlin", arrays, reference, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(root / f"invoke-{index}")
    assert _record(root / "merlin/host/model.so") == library


def test_actual_serial_images_keep_word_state_and_zero_trip_final_positions(compiled):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    root, transpose, dtype = compiled
    for index in (1, 2):
        inputs, expected = arguments(transpose, dtype, index, final_only=True)
        arrays, reference = root / f"target-arguments-{index}.npz", root / f"target-reference-{index}.npz"
        np.savez(arrays, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=arrays, reference=reference, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
