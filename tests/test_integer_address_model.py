"""Complete models keep paired address control, dependent state and every result."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_integer_address_regions import capture
from test_pointer_addresses import REQUIRED

pytestmark = pytest.mark.skipif(not REQUIRED or not os.environ.get("T2M_MODEL2MLIR_PYTHON"),
                                reason="existing model2MLIR/native/Merlin tools required")


@pytest.mark.parametrize("width,mode", [(32, "for"), (64, "while"), (32, "branch"), (64, "alternating")])
def test_complete_model_retains_original_address_regions_and_state_snapshots(tmp_path, width, mode):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle

    parent = tmp_path / "parent"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_integer_address_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(parent),
                              "--width", str(width), "--mode", mode], capture_output=True, text=True,
                             timeout=60, check=False,
                             env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stdout + process.stderr
    implementation = capture(tmp_path, mode, width)
    names = ("n", "capacity", "steps", "origin", "stride", "choice")
    bindings = [KernelBinding(f"t2m_demo::integer_address_{mode}_{width}", implementation,
        operands={"x": "x", "y": "y", "state": "initial"},
        scalar_arguments=dict(zip(names, values, strict=True)),
        runtime_arguments={name: name for name in names}) for values in (
            (19, 128, 4, 80, -3, 1), (19, 128, 2, 31, -8, -1),
            (0, 0, 3, 256, 8, -2), (19, 0, 0, -256, -8, 0))]
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
        assert expected.files == [f"out{i}" for i in range(11)]
        for path in (tmp_path / "qualification/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == expected.files
                for name in expected.files:
                    assert (actual[name].shape, actual[name].dtype, actual[name].tobytes()) == (
                        expected[name].shape, expected[name].dtype, expected[name].tobytes())
    from test_spike_model import TOOLS
    if TOOLS:
        from scripts import qualify_spike_model as target
        options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
        report = target.qualify(out, tmp_path / "target", comparison="words", **options)
        assert target.verify(tmp_path / "target", **options) == report
        assert report["evaluation"]["result_count"] == 11
