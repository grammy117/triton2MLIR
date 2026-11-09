import json
import math
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
CLANG = os.environ.get("T2M_MERLIN_CLANG")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and M2M), reason="existing shared/native/model2MLIR environments required")


@pytest.fixture(scope="module", params=["exact-tiles", "ragged-tiles", "block-pointers", "tensor-descriptors"])
def model_inputs(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("actual-model2mlir")
    script = Path(__file__).resolve().parents[1] / "scripts/capture_model_binding_demo.py"
    block_pointers = request.param == "block-pointers"
    tensor_descriptors = request.param == "tensor-descriptors"
    ragged = request.param != "exact-tiles"
    flags = ["--tensor-descriptors"] if tensor_descriptors else ["--block-pointers"] if block_pointers else ["--ragged"] if ragged else []
    process = subprocess.run([M2M, str(script), "--out", str(root / "capture"), *flags],
                             capture_output=True, text=True, timeout=60,
                             env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert process.returncode == 0, process.stderr
    from t2m.examples.tensor_kernels import loop_matmul, ragged_matmul, blockptr_matmul, descriptor_matmul
    kernel = descriptor_matmul if tensor_descriptors else blockptr_matmul if block_pointers else ragged_matmul if ragged else loop_matmul
    m, n, k = (35, 20, 36) if tensor_descriptors else (35, 21, 37) if ragged else (32, 32, 32)
    constants = {"N": n, "K": k, "BM": 16, "BN": 16, "BK": 16}
    if ragged:
        constants.update(M=m)
        if not (block_pointers or tensor_descriptors):
            constants.update(TRANSPOSED=False)
    source = root / "kernel.ttir"
    source.write_text(capture_ttir(kernel, {**{n: "*fp32" for n in ("x", "y", "out")},
                                              **{n: "constexpr" for n in constants}}, constants))
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", (m, k)), TensorBuffer("y", (k, n)),
                                    TensorBuffer("out", (m, n), "write")), grid=((m + 15) // 16, (n + 15) // 16))
    result = convert_shared(source, abi, root / "implementation", optimizer=OPT)
    return root / "capture", KernelBinding("t2m_demo::gemm", result, scalar_arguments={"alpha": 1})


def test_actual_triton_op_model_binds_parameters_and_merlin_interface(tmp_path, model_inputs):
    capture, binding = model_inputs
    out = tmp_path / "bound"
    write_model_bundle(capture, [binding], out, work=tmp_path / "work", python=NATIVE,
                       merlin_root=MERLIN, merlin_python=os.environ.get("T2M_MERLIN_PYTHON"),
                       merlin_host=bool(MERLIN and CLANG), merlin_clang=CLANG, rtol=1e-5, atol=5e-6)
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
        assert consumer["compiler_input_staging"] == "passed"
        assert "consumer/compiler/source-correspondence.json" in receipt["artifacts"]
        assert (out / "consumer/runtime/model_call.c").is_file()
        assert consumer["target_execution"] == "not-performed"
        if CLANG:
            host = json.loads((out / "consumer/host/execution.json").read_text())
            assert host["host_execution"] == host["whole_body_lowering"] == "passed"
            assert host["evaluation"]["invocations"] == 2 and host["evaluation"]["functional_inputs_preserved"]
            assert "ownership-based-buffer-deallocation" in host["pipeline"]
            receipt_bytes = (out / "capture_receipt.json").read_bytes()
            missing = json.loads(receipt_bytes)
            missing["artifacts"].pop("consumer/host/execution.json")
            (out / "capture_receipt.json").write_text(json.dumps(missing))
            with pytest.raises(ValueError, match="success lacks"):
                verify_bundle(out)
            (out / "capture_receipt.json").write_bytes(receipt_bytes)
    (out / "kernels/0/source.ttir").write_text("changed")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_bundle(out)


@pytest.mark.parametrize("profile", ["buffer", "guarded-buffer"])
def test_mixed_memory_kernel_binds_real_model_and_preserves_parent_data(tmp_path, model_inputs, profile):
    capture, original = model_inputs
    source = tmp_path / "source.ttir"
    source.write_text(original.implementation.source_ttir)
    implementation = convert_shared(source, original.implementation.spec, tmp_path / "implementation", optimizer=OPT, memory_profile=profile)
    binding = KernelBinding(original.qualname, implementation, operands=original.operands,
                            results=original.results, scalar_arguments=original.scalar_arguments)
    out = tmp_path / "bound"
    write_model_bundle(capture, [binding], out, work=tmp_path / "work", python=NATIVE,
        merlin_root=MERLIN, merlin_python=os.environ.get("T2M_MERLIN_PYTHON"),
        merlin_host=bool(MERLIN and CLANG), merlin_clang=CLANG, rtol=1e-5, atol=6e-6)
    receipt = verify_bundle(out)
    evidence = "memory-admission.json" if profile == "guarded-buffer" else "memory-witness.mlir"
    assert "kernels/0/" + evidence in receipt["artifacts"]
    for name in ("weights.safetensors", "inputs.npz", "golden.npy", "extra.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    if MERLIN:
        assert (out / "consumer/compiler/source/kernels/0/memory-stage.mlir").is_file()
        if profile == "guarded-buffer":
            signature = json.loads((out / "consumer/compiler/signature.json").read_text())
            expected = 16 * sum(math.prod(b.shape) for b in implementation.spec.outputs)
            assert signature["guard_storage_bytes"] == expected


def test_parent_integrity_failure_prevents_any_model_publication(tmp_path, model_inputs):
    capture, binding = model_inputs
    corrupted = tmp_path / "corrupted"
    shutil.copytree(capture, corrupted)
    (corrupted / "golden.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="parent capture integrity mismatch"):
        write_model_bundle(corrupted, [binding], tmp_path / "out", work=tmp_path / "work", python=NATIVE)
    assert not (tmp_path / "out").exists() and not (tmp_path / "work").exists()


@pytest.mark.skipif(not MERLIN, reason="selected Merlin checkout required")
def test_consumer_failure_cannot_leave_a_successful_outer_receipt(tmp_path, model_inputs, monkeypatch):
    capture, binding = model_inputs

    def fail(*args, **kwargs):
        raise ValueError("explicit consumer failure")

    monkeypatch.setattr("t2m.model_bundle.check_merlin_host", fail)
    with pytest.raises(ValueError, match="explicit consumer failure"):
        write_model_bundle(capture, [binding], tmp_path / "out", work=tmp_path / "work", python=NATIVE,
                           merlin_root=MERLIN, merlin_python=os.environ.get("T2M_MERLIN_PYTHON"),
                           merlin_host=True, rtol=1e-5, atol=5e-6)
    assert not (tmp_path / "out/capture_receipt.json").exists()
