"""True model2MLIR captures retain two independently bound selected-pointer calls."""

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

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
M2M, MERLIN = os.environ.get("T2M_MODEL2MLIR_PYTHON"), os.environ.get("T2M_MERLIN_ROOT")
CLANG = os.environ.get("T2M_MERLIN_CLANG")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and M2M), reason="existing shared/native/model2MLIR environments required")


@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_real_model_retains_selected_pointers_weights_and_two_call_bindings(tmp_path, dtype):
    from t2m.examples.selected_pointer_kernels import selected_permute
    script = Path(__file__).resolve().parents[1] / "scripts/capture_selected_memory_model.py"
    capture = tmp_path / "capture"
    process = subprocess.run([M2M, str(script), "--out", str(capture), "--dtype", dtype],
        capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr
    bindings = []
    for choice in (0, 1):
        constants = {"n": 13, "m": 29, "choice": choice, "B": 8, "MODE": 2}
        source = tmp_path / f"kernel{choice}.ttir"
        pointer = "*fp32" if dtype == "f32" else "*fp16"
        source.write_text(capture_ttir(selected_permute, {"x": pointer, "y": pointer, "out": pointer,
            **{name: "constexpr" for name in constants}}, constants))
        abi = TensorABI("selected_permute", (TensorBuffer("x", (29,), dtype=dtype), TensorBuffer("y", (29,), dtype=dtype),
            TensorBuffer("out", (13,), "write", dtype=dtype)))
        result = convert_shared(source, abi, tmp_path / f"translation{choice}", optimizer=OPT,
                                memory_profile="guarded-buffer", address_policy="source-words")
        bindings.append(KernelBinding("t2m_demo::selected_gather", result, scalar_arguments={"choice": choice}))
    out = tmp_path / "model"
    write_model_bundle(capture, bindings, out, work=tmp_path / "qualification", python=NATIVE,
        merlin_root=MERLIN, merlin_python=os.environ.get("T2M_MERLIN_PYTHON"),
        merlin_host=bool(MERLIN and CLANG), merlin_clang=CLANG, rtol=1e-5, atol=3e-6)
    receipt = verify_bundle(out)
    assert receipt["schema"] == "t2m.bound-model-receipt.v1"
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "input_order.json", "golden.npy", "extra.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    calls = json.loads((out / "kernel-bindings.json").read_text())["calls"]
    assert len(calls) == 2
    profiles = json.loads((out / "compatibility.json").read_text())["kernel_profiles"]
    for i in (0, 1):
        assert f"kernels/{i}/word-input.mlir" in receipt["artifacts"]
        trace = json.loads((out / f"kernels/{i}/provenance.json").read_text())
        assert any(r["opcode"] == "tt.cat" for r in trace["source_operations"])
        assert any(t["kind"] == "selected-pointer-memory" for t in trace["transformations"])
        assert any(t["kind"] == "ttir-tile-gather" for t in trace["transformations"])
        assert profiles[f"kernels/{i}"]["profile"] == "shared-source-word-buffer"
        differences = "\n".join(profiles[f"kernels/{i}"]["differences"])
        assert "finite mixed-base pointers" in differences and "TTIR cat" in differences
    if MERLIN:
        consumer = json.loads((out / "consumer/interface.json").read_text())
        assert consumer["compiler_input_staging"] == consumer["interface_ingestion"] == "passed"
        if CLANG:
            host = json.loads((out / "consumer/host/execution.json").read_text())
            assert host["host_execution"] == host["reference_comparison"] == "passed"
            assert host["evaluation"]["functional_inputs_preserved"]
