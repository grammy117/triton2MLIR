import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from t2m.bundle import verify_bundle
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT = os.environ.get("TRITON_SHARED_OPT")
NATIVE = os.environ.get("T2M_NATIVE_PYTHON")
M2M = os.environ.get("T2M_MODEL2MLIR_PYTHON")
MERLIN = os.environ.get("T2M_MERLIN_ROOT")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and M2M), reason="existing shared/native/model2MLIR environments required")


@pytest.fixture(scope="module")
def model_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("actual-model2mlir")
    script = Path(__file__).resolve().parents[1] / "scripts/capture_model_binding_demo.py"
    process = subprocess.run([M2M, str(script), "--out", str(root / "capture")],
                             capture_output=True, text=True, timeout=60,
                             env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert process.returncode == 0, process.stderr
    from t2m.examples.tensor_kernels import loop_matmul
    constants = {"N": 32, "K": 32, "BM": 16, "BN": 16, "BK": 16}
    source = root / "kernel.ttir"
    source.write_text(capture_ttir(loop_matmul, {**{n: "*fp32" for n in ("x", "y", "out")},
                                              **{n: "constexpr" for n in constants}}, constants))
    abi = TensorABI("loop_matmul", (TensorBuffer("x", (32, 32)), TensorBuffer("y", (32, 32)),
                                   TensorBuffer("out", (32, 32), "write")), grid=(2, 2))
    result = convert_shared(source, abi, root / "implementation", optimizer=OPT)
    return root / "capture", KernelBinding("t2m_demo::gemm", result, scalar_arguments={"alpha": 1})


def test_actual_triton_op_model_binds_parameters_and_merlin_interface(tmp_path, model_inputs):
    capture, binding = model_inputs
    out = tmp_path / "bound"
    write_model_bundle(capture, [binding], out, work=tmp_path / "work", python=NATIVE,
                       merlin_root=MERLIN, merlin_python=os.environ.get("T2M_MERLIN_PYTHON"), rtol=1e-5, atol=5e-6)
    receipt = verify_bundle(out)
    assert receipt["schema"] == "t2m.bound-model-receipt.v1"
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "input_order.json", "extra.npz", "golden.npy"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    assert (out / "parent/capture_receipt.json").read_bytes() == (capture / "capture_receipt.json").read_bytes()
    assert not (out / "frontend-trace.json").exists()  # The parent trace binds the parent IR.
    execution = json.loads((out / "execution.json").read_text())
    assert execution["reference_comparison"] == "passed"
    assert [b["storage"] for b in execution["materialized_arguments"]] == ["weights:weight", "inputs:in0"]
    if MERLIN:
        consumer = json.loads((out / "consumer/interface.json").read_text())
        assert consumer["interface_ingestion"] == consumer["runtime_generation"] == "passed"
        assert (out / "consumer/runtime/model_call.c").is_file()
        assert consumer["target_execution"] == "not-performed"
    (out / "kernels/0/source.ttir").write_text("changed")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_bundle(out)


def test_parent_integrity_failure_prevents_any_model_publication(tmp_path, model_inputs):
    capture, binding = model_inputs
    corrupted = tmp_path / "corrupted"
    shutil.copytree(capture, corrupted)
    (corrupted / "golden.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="parent capture integrity mismatch"):
        write_model_bundle(corrupted, [binding], tmp_path / "out", work=tmp_path / "work", python=NATIVE)
    assert not (tmp_path / "out").exists() and not (tmp_path / "work").exists()
