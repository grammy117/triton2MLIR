"""Complete models preserve volatile program phases and all source results."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import test_barrier_models as common
from test_volatile_barriers import implementation

from t2m.bundle import _record
from t2m.consumer import invoke_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle

pytestmark = pytest.mark.skipif(not common.TOOLS, reason="existing source/model/native/Merlin tools required")


@pytest.fixture(scope="module", params=["i64", "f32"])
def selected(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("volatile-phase-model-" + request.param)
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_barrier_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture),
        "--dtype", request.param, "--volatile"], capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    kernel_root = root / "kernel"
    kernel_root.mkdir()
    kernel = implementation(kernel_root, request.param)
    bindings = [KernelBinding("t2m_demo::program_exchange", kernel, operands={"state": "x"}, results=("state", "out"),
        scalar_arguments={"steps": steps, **({"n": 0} if empty else {})},
        runtime_arguments={"steps": "steps", **({"n": "n"} if empty else {})},
        runtime_operands={} if empty else {"n": "n"}) for steps, empty in ((2, False), (3, False), (4, True))]
    package = write_model_bundle(capture, bindings, root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, request.param, capture, package


def invocation(root, dtype, n, index):
    dt = np.int64 if dtype == "i64" else np.float32
    x = np.arange(n * 4, dtype=dt).reshape(n, 4) + index
    if dtype == "i64":
        x += 1 << 55
    bias = np.asarray([index + 2, -9, 11, -4], dtype=dt)
    inputs = root / f"inputs-{index}.npz"
    np.savez(inputs, in0=bias, in1=x)
    return inputs, common.reference(x, bias)


def test_complete_volatile_phase_model_retains_every_original_result(selected):
    root, _, capture, package = selected
    common.test_complete_model_retains_source_phases_snapshots_and_every_result((root, capture, package))


def test_changed_volatile_phase_model_shapes_and_parameters_reuse_the_original_library(selected):
    root, dtype, _, package = selected
    library = _record(package / "consumer/host/model.so")
    for index, n in enumerate((2, 19, 11)):
        inputs, expected = invocation(root, dtype, n, index)
        actual, observation = invoke_merlin_host(package / "consumer", inputs, root / f"replay-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert observation["functional_inputs_preserved"]
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def test_rehashed_source_effect_and_phase_claims_refuse_both_complete_model_readers(selected):
    root, _, capture, package = selected
    common.test_rehashed_phase_claims_and_payload_refuse_both_readers((root, capture, package))


def test_original_volatile_phase_model_shape_domain_still_aborts(selected):
    import signal
    from t2m.consumer import observe_merlin_host

    root, dtype, _, package = selected
    inputs, _ = invocation(root, dtype, 1, 50)
    work = root / "domain-abort"
    outcome = observe_merlin_host(package / "consumer", inputs, work,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert outcome["returncode"] == -signal.SIGABRT and not outcome["timed_out"]
    assert not (work / "outputs.npz").exists()


def test_actual_serial_target_retains_all_volatile_phase_model_results(selected):
    import sys
    from test_spike_model import TOOLS as TARGET_TOOLS, Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, dtype, _, package = selected
    inputs, expected = invocation(root, dtype, 19, 8)
    reference = root / "target-reference.npz"
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        report = Q.qualify(package, root / "target", result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"], inputs=inputs, reference=reference)
        assert report["evaluation"]["result_count"] == 9
    finally:
        sys.path.remove(scripts)
