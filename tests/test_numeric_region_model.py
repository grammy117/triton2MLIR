"""Complete models retain coarse numeric bodies, parameters and every result."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_numeric_regions import capture
from test_tensor_regions import REQUIRED

pytestmark = pytest.mark.skipif(
    not REQUIRED or not os.environ.get("T2M_MODEL2MLIR_PYTHON"),
    reason="existing native/model2MLIR/Merlin tools required",
)


@pytest.mark.parametrize("dtype", ["f32", "i64"])
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_numeric_callbacks_bind_dependent_models_and_all_original_results(tmp_path, dtype, profile):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle

    script = Path(__file__).resolve().parents[1] / "scripts/capture_numeric_region_model.py"
    parent = tmp_path / "parent"
    process = subprocess.run(
        [os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(parent), "--dtype", dtype],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)),
    )
    assert process.returncode == 0, process.stdout + process.stderr
    forms = (("dot", 2), ("scaled-dot", 3), ("dot-zero", 4)) if dtype == "f32" else (
        ("histogram", 0), ("gather", 1))
    bindings = []
    for family, kind in forms:
        root = tmp_path / f"kernel-{kind}"
        root.mkdir()
        implementation = capture(root, family, dtype, profile)
        for n in ((19, 0) if kind in (0, 2) else (19,)):
            bindings.append(KernelBinding("t2m_demo::numeric_regions", implementation,
                scalar_arguments={"n": n, "kind": kind}, runtime_arguments={"n": "n"}))
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
    count = 7 if dtype == "f32" else 6
    with np.load(parent / "goldens.npz", allow_pickle=False) as expected:
        for path in (tmp_path / "qualification/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == expected.files == [f"out{i}" for i in range(count)]
                for name in expected.files:
                    assert actual[name].shape == expected[name].shape and actual[name].dtype == expected[name].dtype
                    assert actual[name].tobytes() == expected[name].tobytes()
    from test_spike_model import TOOLS
    if TOOLS:
        from scripts import qualify_spike_model as target
        options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
        report = target.qualify(out, tmp_path / "target", comparison="words", **options)
        assert target.verify(tmp_path / "target", **options) == report
        assert report["evaluation"]["result_count"] == count
