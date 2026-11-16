"""Relative-pointer models keep every original value, byte distance and snapshot."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_pointer_addresses import REQUIRED
from test_pointer_differences import capture

pytestmark = pytest.mark.skipif(
    not REQUIRED or not os.environ.get("T2M_MODEL2MLIR_PYTHON"),
    reason="existing native/model2MLIR/Merlin tools required",
)


@pytest.mark.parametrize("width", [32, 64])
def test_complete_model_retains_relative_byte_results_and_dependent_runtime_views(tmp_path, width):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle

    parent = tmp_path / "parent"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_pointer_difference_model.py"
    process = subprocess.run(
        [os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(parent), "--width", str(width)],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)),
    )
    assert process.returncode == 0, process.stdout + process.stderr
    implementation = capture(tmp_path, width)
    names = ("n", "origin", "stride", "capacity", "steps")
    bindings = [KernelBinding("t2m_demo::pointer_difference", implementation,
        scalar_arguments=dict(zip(names, values, strict=True)), runtime_arguments=dict(zip(names, names, strict=True)))
        for values in ((19, 128, 3, 256, 3), (7, 9, 1, 19, 2), (0, 256, -8, 0, 4), (1, 256, -8, 256, 0))]
    out = tmp_path / "bound"
    write_model_bundle(parent, bindings, out, work=tmp_path / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_host=True,
        merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(out)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz",
                 "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (parent / name).read_bytes()
    with np.load(parent / "goldens.npz", allow_pickle=False) as expected:
        for path in (tmp_path / "qualification/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == expected.files == [f"out{i}" for i in range(10)]
                for name in expected.files:
                    assert actual[name].shape == expected[name].shape and actual[name].dtype == expected[name].dtype
                    assert actual[name].tobytes() == expected[name].tobytes()
    from test_spike_model import TOOLS
    if TOOLS:
        from scripts import qualify_spike_model as target
        options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
        report = target.qualify(out, tmp_path / "target", comparison="words", **options)
        assert target.verify(tmp_path / "target", **options) == report
        assert report["evaluation"]["result_count"] == 10
