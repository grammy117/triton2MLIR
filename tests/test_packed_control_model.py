"""Real model calls retain packed scalar algorithms through host and target."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_packed_control import capture as translate
from test_spike_model import Q, TOOLS as TARGET_TOOLS


TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


@pytest.mark.skipif(not TOOLS, reason="existing model2MLIR, native and Merlin tools required")
@pytest.mark.parametrize("dtype,pack,while_loop", [("i64", 4, True), ("f32", 2, False)])
def test_learned_model_retains_packed_loops_counts_and_all_four_results(tmp_path, dtype, pack, while_loop):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY

    source = tmp_path / "source"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_packed_control_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(source), "--dtype", dtype,
        "--pack", str(pack), *([] if while_loop else ["--for-loop"])], capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stdout + process.stderr
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    result = translate(kernel, pack, dtype, while_loop, "guarded-buffer")
    bindings = [KernelBinding("t2m_demo::packed_control", result, tensor_views=POLICY,
        operands={name: name for name in ("x", "y", "counts")}, scalar_arguments={"n": 17, "iterations": iterations},
        runtime_arguments={name: name for name in ("n", "iterations")}) for iterations in (2, 0)]
    out = tmp_path / "bound"
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    write_model_bundle(source, bindings, out, work=tmp_path / "host", python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=root, merlin_python=python, merlin_host=True, merlin_clang=clang)
    verify_bundle(out)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (source / name).read_bytes()
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert len(trace["calls"]) == 2 and len(trace["kernels"]) == 1 and trace["model_abi"] == "unchanged"
    assert "prov.map_pack" in (out / "model.mlir").read_text()
    assert ("scf.while" if while_loop else "scf.for") in (out / "model.mlir").read_text()
    with np.load(source / "goldens.npz", allow_pickle=False) as expected:
        for path in (tmp_path / "host/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == [f"out{i}" for i in range(4)]
                for key in actual.files:
                    assert (actual[key].shape, actual[key].dtype, actual[key].tobytes()) == (
                        expected[key].shape, expected[key].dtype, expected[key].tobytes())
    if TARGET_TOOLS:
        report = Q.qualify(out, tmp_path / "target", merlin_root=root, python=python, comparison="words")
        assert report["evaluation"]["result_count"] == 4
        assert Q.verify(tmp_path / "target", merlin_root=root, python=python) == report
