"""Selected-root and while kernels preserve complete dependent-call models."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_checked_memory_continuation import KINDS, capture, reference

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host, observe_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle

TOOLS = all(os.environ.get(k) for k in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON",
    "T2M_MODEL2MLIR_PYTHON", "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/model/Merlin tools required")


@pytest.fixture(scope="module", params=KINDS)
def selected(tmp_path_factory, request):
    root, kind = tmp_path_factory.mktemp(request.param + "-model"), request.param
    parent = root / "parent"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_root_state_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(parent), "--kind", kind],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    result = capture(root, kind, static_width=17, control_dtype="i64")
    binding = KernelBinding("t2m_demo::root_state_" + kind, result, results=("out", "last"),
        runtime_operands={name: name for name in ("lower", "upper", "stride", "memory_stride")})
    package = write_model_bundle(parent, [binding], root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, parent, package, kind


def test_every_result_original_parent_and_empty_observation_survives(selected):
    _root, parent, package, _kind = selected
    receipt = verify_bundle(package)
    for source, destination in receipt["parent_artifact_bindings"].items():
        assert (parent / source).read_bytes() == (package / destination).read_bytes()
    with np.load(parent / "goldens.npz", allow_pickle=False) as expected, np.load(
            package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(actual.files) == len(expected.files) == 13
        for name in expected.files:
            assert (actual[name].shape, actual[name].dtype, actual[name].tobytes()) == (
                expected[name].shape, expected[name].dtype, expected[name].tobytes())
        assert actual["out12"].size == 0


def invocation(root, package, kind, index):
    controls = [(0, 8, 1, -2), (3, 1, 2, 2), (0, 0, 1, 0)][index]
    x = np.arange(64*17, dtype=np.int64).reshape(64, 17)*(37+index) + (1 << 58)
    x[1::2] *= -1
    weight = np.arange(64*17, dtype=np.int64).reshape(64, 17)*(13+index) + (1 << 54)
    table = -np.arange(32*17, dtype=np.int64).reshape(32, 17)*(19+index) - (1 << 55)
    first, earlier = reference(kind, x, table, controls)
    shifted = first + weight
    second, final = reference(kind, shifted, table, controls)
    scalars = [np.asarray(v, np.int64) for v in controls]
    expected = (second, first, earlier, final, x, shifted, weight, table, *scalars, first[:0])
    values = dict(zip(("lower", "upper", "stride", "memory_stride"), scalars, strict=True),
        x=x, weight=weight, table=table)
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    inputs, golden = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
    np.savez(inputs, **{f"in{i}": values[row.get("weight") or row["name"]]
                       for i, row in enumerate(manifest.values())})
    np.savez(golden, **{f"out{i}": value for i, value in enumerate(expected)})
    return inputs, golden, expected


def test_one_model_library_reuses_both_calls_and_every_changed_input(selected):
    root, _parent, package, kind = selected
    library = _record(package / "consumer/host/model.so")
    for index in range(3):
        inputs, _golden, expected = invocation(root, package, kind, index)
        actual, report = invoke_merlin_host(package / "consumer", inputs, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        assert len(actual) == len(expected)
        for got, wanted in zip(actual, expected, strict=True):
            assert (got.shape, got.dtype, got.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def test_original_control_domain_still_aborts_without_model_results(selected):
    root, _parent, package, kind = selected
    inputs, _golden, _expected = invocation(root, package, kind, 0)
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    with np.load(inputs, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    position = next(i for i, row in enumerate(manifest.values()) if row.get("name") == "memory_stride")
    arrays[f"in{position}"] = np.asarray(3, np.int64)
    invalid = root / "invalid.npz"
    np.savez(invalid, **arrays)
    observed = root / "invalid-observation"
    report = observe_merlin_host(package / "consumer", invalid, observed,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert report["returncode"] != 0
    assert not (observed / "outputs.npz").exists()


def test_actual_serial_model_images_retain_all_thirteen_results(selected, monkeypatch):
    from test_spike_model import TOOLS as SERIAL_TOOLS
    from test_spike_model import Q

    if not SERIAL_TOOLS:
        pytest.skip("existing cross compiler and Spike required")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    root, _parent, package, kind = selected
    for index in (0, 2):
        inputs, golden, _expected = invocation(root, package, kind, index)
        target = root / f"target-{index}"
        report = Q.qualify(package, target, result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
            inputs=inputs, reference=golden)
        assert report["evaluation"]["result_count"] == 13
        assert Q.verify(target, merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"]) == report
