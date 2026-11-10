"""Real model2MLIR captures bind selected integer and floating storage."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

OPT, NATIVE, M2M = (os.environ.get(n) for n in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON"))
MERLIN, CLANG = os.environ.get("T2M_MERLIN_ROOT"), os.environ.get("T2M_MERLIN_CLANG")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and M2M and MERLIN and CLANG), reason="existing shared/model2MLIR/Merlin required")


@pytest.mark.parametrize("width", [16, 32, 64])
def test_mixed_view_model_preserves_weights_initial_state_and_raw_results(tmp_path, width):
    from t2m.bundle import verify_bundle
    from t2m.examples.pointer_view_kernels import word_view_memory
    from t2m.frontend import capture_ttir
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    capture = tmp_path / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_mixed_view_model.py"
    process = subprocess.run([M2M, str(script), "--out", str(capture), "--width", str(width)], capture_output=True,
        text=True, timeout=60, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr
    constants = {"n": 13, "steps": 2, "B": 8, "WIDTH": width, "SELECT": True, "BEFORE": True, "WHILE": True, "SCALAR": False}
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(word_view_memory, {"x": f"*i{width}", "y": f"*fp{width}", "out": f"*fp{width}",
        **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("word_view_memory", (TensorBuffer("x", (4, 13), dtype=f"i{width}"), TensorBuffer("y", (4, 13), dtype=f"f{width}"),
        TensorBuffer("out", (4, 13), "readwrite", dtype=f"f{width}")), grid=(2,))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer", address_policy="source-words")
    binding = KernelBinding("t2m_demo::mixed_word_view", result, operands={"x": "x", "y": "state", "out": "initial"})
    out = tmp_path / "model"
    write_model_bundle(capture, [binding], out, work=tmp_path / "qualification", python=NATIVE,
        merlin_root=MERLIN, merlin_python=os.environ.get("T2M_MERLIN_PYTHON"), merlin_host=True, merlin_clang=CLANG)
    verify_bundle(out)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "extra.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    expected = np.load(capture / "golden.npy", allow_pickle=False).view(getattr(np, f"uint{width}"))
    for path in (tmp_path / "qualification/native/outputs.npz", out / "consumer/host/outputs.npz"):
        with np.load(path, allow_pickle=False) as actual:
            np.testing.assert_array_equal(actual["out0"].view(getattr(np, f"uint{width}")), expected)
    host = json.loads((out / "consumer/host/execution.json").read_text())
    assert host["host_execution"] == host["reference_comparison"] == "passed"
    assert host["evaluation"]["functional_inputs_preserved"]
