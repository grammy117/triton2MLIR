"""Model2MLIR packages retain packed source computation and every original result."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_packed_source_assembly import PROVIDER, TOOLS, capture

pytestmark = pytest.mark.skipif(not (TOOLS and PROVIDER and os.environ.get("T2M_MODEL2MLIR_PYTHON")),
                                reason="existing source/model/native/Merlin tools required")


@pytest.fixture(scope="module", params=((32, 2), (64, 4)))
def compiled(tmp_path_factory, request):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY

    width, pack = request.param
    root = tmp_path_factory.mktemp(f"packed-assembly-model-i{width}-{pack}")
    repo = Path(__file__).resolve().parents[1]
    parent = root / "parent"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"],
        str(repo / "scripts/capture_packed_assembly_model.py"), "--out", str(parent),
        "--dtype", f"i{width}", "--pack", str(pack)], capture_output=True, text=True,
        timeout=60, check=False, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert process.returncode == 0, process.stdout + process.stderr
    result = capture(root / "source", width, pack)
    bindings = [KernelBinding("t2m_demo::packed_assembly_state", result, tensor_views=POLICY,
        operands={"x": "x", "y": "y"}, scalar_arguments={"n": n, "steps": steps},
        runtime_arguments={"n": "n", "steps": "steps"}) for n, steps in ((32, 3), (32, 2), (0, 3))]
    out = root / "bound"
    write_model_bundle(parent, bindings, out, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"], merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_host=True, merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(out)
    return root


def test_original_calls_weights_snapshots_and_all_nine_results_remain(compiled):
    root = compiled
    parent, out = root / "parent", root / "bound"
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz",
                 "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (parent / name).read_bytes()
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert len(trace["calls"]) == 3 and len(trace["kernels"]) == 1 and trace["model_abi"] == "unchanged"
    for actual_path in (root / "native/native/outputs.npz", out / "consumer/host/outputs.npz"):
        with np.load(parent / "goldens.npz", allow_pickle=False) as expected, np.load(actual_path, allow_pickle=False) as actual:
            assert actual.files == [f"out{i}" for i in range(9)]
            for name in actual.files:
                assert (actual[name].shape, actual[name].dtype, actual[name].tobytes()) == (
                    expected[name].shape, expected[name].dtype, expected[name].tobytes())


def test_actual_serial_model_preserves_all_nine_original_results(compiled):
    from test_spike_model import TOOLS as SERIAL_TOOLS
    from test_spike_model import Q

    if not SERIAL_TOOLS:
        pytest.skip("existing cross compiler and Spike required")
    root = compiled
    report = Q.qualify(root / "bound", root / "target", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], comparison="words")
    assert report["evaluation"]["result_count"] == 9
    assert Q.verify(root / "target", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"]) == report
