"""Whole PyTorch models retain both calls, every result and live memory strides."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_invariant_stride_kernels import arguments, capture, reference

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host, observe_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle

TOOLS = all(os.environ.get(k) for k in ("TRITON_SHARED_OPT", "T2M_MODEL2MLIR_PYTHON", "T2M_MERLIN_ROOT",
    "T2M_MERLIN_PYTHON", "T2M_NATIVE_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing model/source/Merlin tools required")


@pytest.fixture(scope="module", params=("memory", "pure", "mixed", "late",
    "conditional-scalar", "conditional-pointer", "conditional-pure", "conditional-observe", "conditional-nested", "conditional-bit"))
def selected(tmp_path_factory, request):
    conditional = request.param.startswith("conditional-")
    mode = request.param.removeprefix("conditional-")
    root = tmp_path_factory.mktemp(f"invariant-stride-{request.param}-model")
    parent = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_invariant_stride_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(parent),
                          "--conditional-mode" if conditional else "--mode", mode],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    result = capture(root, mode, dtype="i64", conditional_mode=mode if conditional else None)
    binding = KernelBinding("t2m_demo::conditional_stride" if conditional else "t2m_demo::invariant_stride", result, results=("out", "last"),
        runtime_operands={name: name for name in ("lower", "upper", "stride", "memory_stride")})
    package = write_model_bundle(parent, [binding], root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, parent, package, mode, conditional


def test_all_original_model_results_and_parent_bytes_survive(selected):
    _root, parent, package, _mode, _conditional = selected
    receipt = verify_bundle(package)
    for source, destination in receipt["parent_artifact_bindings"].items():
        assert (parent / source).read_bytes() == (package / destination).read_bytes()
    with np.load(parent / "goldens.npz", allow_pickle=False) as expected, np.load(
            package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(expected.files) == len(actual.files) == 11
        for key in expected.files:
            assert (actual[key].shape, actual[key].dtype, actual[key].tobytes()) == (
                expected[key].shape, expected[key].dtype, expected[key].tobytes())


def invocation(root, package, mode, index, *, conditional=False):
    arrays, (first, earlier) = arguments(mode, False, "i64", index, conditional=conditional)
    x, *controls = arrays
    weight = np.arange(64*17, dtype=np.int64).reshape(64, 17) * (13 + index) + (1 << 54)
    shifted = first + weight
    second, final = reference(shifted, controls, mode, conditional=conditional)
    expected = (second, first, earlier, final, x, shifted, weight, *controls)
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    values = dict(zip(("x", "lower", "upper", "stride", "memory_stride"), arrays, strict=True), weight=weight)
    inputs, golden = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
    np.savez(inputs, **{f"in{i}": values[row.get("weight") or row["name"]] for i, row in enumerate(manifest.values())})
    np.savez(golden, **{f"out{i}": value for i, value in enumerate(expected)})
    return inputs, golden, expected


def test_one_library_retains_independent_memory_strides_and_both_original_calls(selected):
    root, _parent, package, mode, conditional = selected
    library = _record(package / "consumer/host/model.so")
    for index in (0, 3, 6):
        inputs, _golden, expected = invocation(root, package, mode, index, conditional=conditional)
        actual, report = invoke_merlin_host(package / "consumer", inputs, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def test_invalid_original_memory_stride_domain_aborts_without_model_results(selected):
    root, _parent, package, mode, conditional = selected
    inputs, _golden, _expected = invocation(root, package, mode, 1, conditional=conditional)
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    with np.load(inputs, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    position = next(i for i, row in enumerate(manifest.values()) if row.get("name") == "memory_stride")
    values[f"in{position}"] = np.asarray(3, np.int64)
    invalid = root / "invalid-memory-stride.npz"
    np.savez(invalid, **values)
    out = root / "domain-abort"
    report = observe_merlin_host(package / "consumer", invalid, out,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert report["returncode"] == -6 and not (out / "outputs.npz").exists()
    (root / "domain-abort.json").write_text(json.dumps(report, indent=2) + "\n")


def test_actual_serial_images_preserve_all_eleven_results_and_zero_trip_division(selected, monkeypatch):
    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial tools required")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    root, _parent, package, mode, conditional = selected
    for index in (0, 3):
        inputs, golden, _expected = invocation(root, package, mode, index, conditional=conditional)
        out = root / f"target-{index}"
        report = Q.qualify(package, out, inputs=inputs, reference=golden, comparison="words", result_mode="execution-owned",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 11
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
