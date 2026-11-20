"""Actual execution ordinals preserve constant and mixed source pointer streams."""

import json
import os

import numpy as np
import pytest
from test_runtime_pointer_steps import TOOLS, arguments, capture

from t2m.diagnostics import ConversionError

pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/native tools required")
MODES = ("memory", "pure", "mixed")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("transpose,dtype", [(t, d) for t in (False, True) for d in ("i32", "i64")])
@pytest.mark.parametrize("profile,policy", [("tensor", "structured"), ("buffer", "structured"),
    ("guarded-buffer", "structured"), ("guarded-buffer", "source-words")])
def test_actual_ordinals_and_final_positions_execute_in_each_profile(tmp_path, mode, transpose, dtype, profile, policy):
    from t2m.native import execute_native

    result = capture(tmp_path, transpose, dtype, profile, policy, ordinal_mode=mode, rows=32)
    if policy == "structured":
        proof = next(row for row in result.trace["transformations"] if row["kind"] == "source-ordinal-pointer-recurrences")
        coefficient = 1 if transpose else 17
        assert [row["iteration_increment"] for row in proof["streams"]] == [coefficient, -coefficient]
        assert [row["step_coefficient"] for row in proof["streams"]] == ([coefficient, -coefficient] if mode == "mixed" else [0, 0])
    if profile == "tensor":
        assert not any(op.name.startswith("memref.") for op in result.module.walk())
        assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
        evidence = result.trace["memory"]["source_loops"][0]
        assert evidence.get("actual_iteration_ordinal_results")
        if mode == "pure":
            assert evidence["memory_effects"] == "no-buffer-accesses"
        else:
            assert evidence["actual_iteration_ordinal_arguments"]
    for index in (1, 2, 3):
        inputs, expected = arguments(transpose, dtype, index, ordinal_mode=mode)
        actual, _ = execute_native(result.mlir_text, inputs, tmp_path / f"native-{index}", python=os.environ["T2M_NATIVE_PYTHON"])
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())


@pytest.mark.parametrize("mode", MODES)
def test_actual_final_pointer_enclosures_still_require_every_root_axis(tmp_path, mode):
    with pytest.raises(ConversionError):
        capture(tmp_path, ordinal_mode=mode, rows=32, bounds=((0, 3), (0, 32), (1, 3)))


@pytest.fixture(scope="module", params=[(m, t, d) for m in MODES for t in (False, True) for d in ("i32", "i64")])
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing Merlin tools required")
    mode, transpose, dtype = request.param
    root = tmp_path_factory.mktemp(f"ordinal-pointer-{mode}-{transpose}-{dtype}")
    result = capture(root, transpose, dtype, ordinal_mode=mode, rows=32)
    inputs, expected = arguments(transpose, dtype, 1, ordinal_mode=mode)
    write_shared_bundle(result, root / "capture", inputs, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    compatibility = json.loads((root / "capture/compatibility.json").read_text())
    assert any("affine source pointer increments" in row for row in compatibility["differences"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, mode, transpose, dtype


def test_one_library_keeps_source_memory_state_and_actual_executed_counts(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    root, mode, transpose, dtype = compiled
    library = _record(root / "merlin/host/model.so")
    for index in range(6):
        inputs, expected = arguments(transpose, dtype, index, ordinal_mode=mode)
        arrays, reference = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
        np.savez(arrays, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        report = invoke_kernel(root / "merlin", arrays, reference, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(root / f"invoke-{index}")
    assert _record(root / "merlin/host/model.so") == library


def test_actual_serial_images_retain_nonunit_control_and_zero_trip_initial_state(compiled):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    root, mode, transpose, dtype = compiled
    for index in (1, 2):
        inputs, expected = arguments(transpose, dtype, index, ordinal_mode=mode)
        arrays, reference = root / f"target-arguments-{index}.npz", root / f"target-reference-{index}.npz"
        np.savez(arrays, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=arrays, reference=reference, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
