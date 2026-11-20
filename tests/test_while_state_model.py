"""Complete PyTorch models preserve while condition effects and old snapshots."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_MODEL2MLIR_PYTHON", "T2M_MERLIN_ROOT",
    "T2M_MERLIN_PYTHON", "T2M_NATIVE_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/model/Merlin tools required")


@pytest.fixture(scope="module")
def selected(tmp_path_factory):
    from test_while_tensor_memory import checked

    root = tmp_path_factory.mktemp("while-state-model")
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_while_state_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture)],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    kernel = root / "kernel"
    kernel.mkdir()
    result = checked(kernel)
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    bindings = [KernelBinding("t2m_demo::while_state", result,
        results=("before", "after", "condition_seen", "state", "checks"),
        scalar_arguments={"limit": limit, "step": step}, runtime_arguments={"limit": "limit", "step": "step"})
        for limit, step in ((5, 2), (9, 3))]
    package = write_model_bundle(capture, bindings, root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, capture, package


def test_complete_original_results_and_parent_archives_survive(selected):
    _root, capture, package = selected
    receipt = verify_bundle(package)
    for source, target in receipt["parent_artifact_bindings"].items():
        assert (capture / source).read_bytes() == (package / target).read_bytes()
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(
            package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(expected.files) == len(actual.files) == 14
        for key in expected.files:
            assert (actual[key].shape, actual[key].dtype, actual[key].tobytes()) == (
                expected[key].shape, expected[key].dtype, expected[key].tobytes())


def invocation(root, package, x_value, counter, name):
    from test_while_tensor_memory import checked_arguments

    x, weight = np.asarray([x_value], np.int64), np.asarray([counter], np.int64)
    initial = np.asarray([min(9, max(0, x_value))], np.int64)
    _, first = checked_arguments(int(initial[0]), counter, 5, 2)
    _, second = checked_arguments(int(first[0][0]), int(first[1][0]), 9, 3)
    wanted = (initial, weight.copy(), *first[2:], first[0], first[1], *second[2:],
              second[0], second[1], x, weight)
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    values = {"x": x, "counter": weight}
    arguments, references = root / f"arguments-{name}.npz", root / f"references-{name}.npz"
    np.savez(arguments, **{f"in{i}": values[row.get("weight") or row["name"]]
                         for i, row in enumerate(manifest.values())})
    np.savez(references, **{f"out{i}": value for i, value in enumerate(wanted)})
    return arguments, references, wanted


def test_one_model_library_retains_changed_state_zero_trips_and_counter_words(selected):
    root, _capture, package = selected
    library = _record(package / "consumer/host/model.so")
    for i, (x, checks) in enumerate(((-7, 19), (7, -9), (12, (1 << 63) - 1), (1, (1 << 56) + 7))):
        arguments, _references, wanted = invocation(root, package, x, checks, str(i))
        actual, report = invoke_merlin_host(package / "consumer", arguments, root / f"invoke-{i}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for observed, expected in zip(actual, wanted, strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (expected.shape, expected.dtype, expected.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def test_actual_serial_model_preserves_every_result_and_final_condition_check(selected, monkeypatch):
    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    root, _capture, package = selected
    for i, (x, checks) in enumerate(((0, (1 << 63) - 1), (9, 31))):
        arguments, references, wanted = invocation(root, package, x, checks, "target-" + str(i))
        out = root / f"target-{i}"
        report = Q.qualify(package, out, result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
            inputs=arguments, reference=references)
        assert report["evaluation"]["result_count"] == len(wanted) == 14
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"],
                        python=os.environ["T2M_MERLIN_PYTHON"]) == report
