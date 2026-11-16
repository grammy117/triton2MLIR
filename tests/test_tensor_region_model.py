"""Complete real models retain local tensor regions, snapshots and all results."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_tensor_regions import REQUIRED, capture

pytestmark = pytest.mark.skipif(
    not REQUIRED or not os.environ.get("T2M_MODEL2MLIR_PYTHON"),
    reason="existing native/model2MLIR/Merlin tools required",
)


@pytest.mark.parametrize("dtype", ["i32", "i64"])
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_complete_model_keeps_tensor_regions_dependent_calls_and_all_results(tmp_path, dtype, profile):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle

    script = Path(__file__).resolve().parents[1] / "scripts/capture_tensor_region_model.py"
    parent = tmp_path / "parent"
    process = subprocess.run(
        [os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(parent), "--dtype", dtype],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)),
    )
    assert process.returncode == 0, process.stdout + process.stderr
    bindings = []
    for kind, mode in (("map", 0), ("reduce-for", 1), ("scan-while-reverse", 2)):
        root = tmp_path / f"kernel-{mode}"
        root.mkdir()
        implementation = capture(root, kind, dtype, profile)
        assert any(item["kind"] == "source-region-tensor-dataflow"
                   for item in implementation.trace["transformations"])
        for n in ((19, 0) if mode == 0 else (19,)):
            bindings.append(KernelBinding("t2m_demo::tensor_regions", implementation,
                scalar_arguments={"n": n, "kind": mode}, runtime_arguments={"n": "n"}))
    out = tmp_path / "bound"
    write_model_bundle(parent, bindings, out, work=tmp_path / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_host=True,
        merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    receipt = verify_bundle(out)
    assert receipt["schema"] == "t2m.bound-model-receipt.v1"
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz",
                 "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (parent / name).read_bytes()
    with np.load(parent / "goldens.npz", allow_pickle=False) as expected:
        for path in (tmp_path / "qualification/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == expected.files == [f"out{i}" for i in range(7)]
                for name in expected.files:
                    assert actual[name].shape == expected[name].shape and actual[name].dtype == expected[name].dtype
                    assert actual[name].tobytes() == expected[name].tobytes()
    from test_spike_model import TOOLS
    if TOOLS:
        from scripts import qualify_spike_model as target
        options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
        report = target.qualify(out, tmp_path / "target", comparison="words", **options)
        assert target.verify(tmp_path / "target", **options) == report
        assert report["evaluation"]["result_count"] == 7
