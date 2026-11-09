"""Region-local event conversion inside a real model2MLIR package."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from t2m.bundle import verify_bundle
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE, M2M = (os.environ.get(n) for n in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON"))
MERLIN, CLANG = os.environ.get("T2M_MERLIN_ROOT"), os.environ.get("T2M_MERLIN_CLANG")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and M2M), reason="existing shared/native/model2MLIR environments required")


def test_real_model_retains_region_events_tile_contractions_and_parent_data(tmp_path):
    from t2m.examples.word_event_kernels import branch_matmul
    capture = tmp_path / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_model_binding_demo.py"
    process = subprocess.run([M2M, str(script), "--out", str(capture)],
        capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr
    constants = {"N": 32, "K": 32, "BM": 16, "BN": 16, "BK": 16}
    path = tmp_path / "source.ttir"
    path.write_text(capture_ttir(branch_matmul, {"x": "*fp32", "y": "*fp32", "out": "*fp32",
        **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("branch_matmul", (TensorBuffer("x", (32, 32)), TensorBuffer("y", (32, 32)),
        TensorBuffer("out", (32, 32), "write")), grid=(2, 2))
    implementation = convert_shared(path, abi, tmp_path / "implementation", optimizer=OPT,
        memory_profile="guarded-buffer", address_policy="source-words")
    assert implementation.trace["memory"]["word_conversion"] == "native-isolated-events-v2"
    assert "linalg.matmul" in implementation.mlir_text and "scf.for" in implementation.mlir_text
    out = tmp_path / "bound"
    write_model_bundle(capture, [KernelBinding("t2m_demo::gemm", implementation, scalar_arguments={"alpha": 1})],
        out, work=tmp_path / "qualification", python=NATIVE, merlin_root=MERLIN,
        merlin_python=os.environ.get("T2M_MERLIN_PYTHON"), merlin_host=bool(MERLIN and CLANG),
        merlin_clang=CLANG, rtol=1e-5, atol=6e-6)
    receipt = verify_bundle(out)
    assert "kernels/0/word-stage.json" in receipt["artifacts"]
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "extra.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    differences = json.loads((out / "compatibility.json").read_text())["kernel_profiles"]["kernels/0"]["differences"]
    assert any("word-stage v2" in difference for difference in differences)
    if MERLIN:
        stage = json.loads((out / "consumer/compiler/source/kernels/0/word-stage.json").read_text())
        assert stage["schema"] == "t2m.source-word-stages.v2"
        if CLANG:
            execution = json.loads((out / "consumer/host/execution.json").read_text())
            assert execution["host_execution"] == execution["reference_comparison"] == "passed"
            assert execution["evaluation"]["functional_inputs_preserved"]
