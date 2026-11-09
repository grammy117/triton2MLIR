"""Qualified mixed-memory packages retain the public ABI and source evidence."""

import json
import os

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.cli import shared_demo
from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.memory_contract import BUFFER_PROFILE, MEMORY_FILES
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared, parse_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


@pytest.mark.parametrize("kernel", ["runtime-prefix", "matrix-add", "ragged-matmul", "descriptor-matmul", "tuple-scan"])
def test_buffer_profile_runs_native_and_merlin_with_exact_memory_evidence(tmp_path, kernel):
    capture = tmp_path / "capture"
    summary = shared_demo(capture, kernel=kernel, optimizer=OPT, python=NATIVE, memory_profile="buffer")
    assert summary["profile"] == BUFFER_PROFILE
    receipt = verify_bundle(capture)
    assert MEMORY_FILES <= receipt["artifacts"].keys()
    module = parse_shared((capture / "model.mlir").read_text())
    module.verify()
    names = {op.name for op in module.walk()}
    assert {"bufferization.to_buffer", "bufferization.to_tensor", "memref.alloc", "scf.forall"} <= names
    assert "linalg.matmul" in names if "matmul" in kernel else "linalg.generic" in names
    assert all(not name.startswith(("tt.", "tts.", "llvm.", "gpu.")) for name in names)
    trace = json.loads((capture / "provenance.json").read_text())
    assert trace["memory"]["semantic_gate"].startswith("source tensor memory analysis")
    assert trace["memory"]["witness_sha256"] == _record(capture / "memory-witness.mlir")["sha256"]
    assert any(op.get("source_ids") or op.get("normalized_ids") for op in trace["emitted_operations"])
    root, python, clang = (os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    if root and python and clang:
        from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
        out = tmp_path / "merlin"
        qualify_kernel(capture, out, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang, rtol=1e-5, atol=6e-6)
        verify_kernel_qualification(out)
        stage = json.loads((out / "compiler/compile-inputs.json").read_text())
        assert stage["payload_profile"] == BUFFER_PROFILE
        assert all("source/" + name in stage["members"] for name in MEMORY_FILES)
        execution = json.loads((out / "host/execution.json").read_text())
        assert execution["evaluation"]["functional_inputs_preserved"]
    # Rehashing an evidence artifact cannot break its relation to provenance.
    raw = (capture / "memory-witness.mlir").read_bytes()
    (capture / "memory-witness.mlir").write_bytes(raw + b"\n")
    receipt["artifacts"]["memory-witness.mlir"] = _record(capture / "memory-witness.mlir")
    (capture / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="memory evidence differs"):
        verify_bundle(capture)


def test_ir_only_cli_keeps_memory_evidence_without_a_numeric_receipt(tmp_path):
    from t2m.cli import main
    from t2m.examples.tensor_kernels import matrix_add
    from t2m.memory_contract import validate_memory_evidence
    source, spec, out = (tmp_path / name for name in ("source.ttir", "abi.json", "ir"))
    source.write_text(capture_ttir(matrix_add, {"x": "*fp32", "y": "*fp32", "out": "*fp32", "M": "constexpr", "N": "constexpr"}, {"M": 4, "N": 8}))
    abi = TensorABI("matrix_add", (TensorBuffer("x", (4, 8)), TensorBuffer("y", (4, 8)), TensorBuffer("out", (4, 8), "write")))
    spec.write_text(json.dumps(abi.as_dict()))
    assert main(["shared-tensor", str(source), "--abi", str(spec), "--out", str(out),
                 "--optimizer", OPT, "--memory-profile", "buffer"]) == 0
    trace = json.loads((out / "provenance.json").read_text())
    assert validate_memory_evidence(trace, {name: (out / name).read_bytes() for name in MEMORY_FILES}) == BUFFER_PROFILE
    assert not (out / "capture_receipt.json").exists()


def test_memory_profile_does_not_bypass_source_ownership_or_initialize_fresh_holes(tmp_path):
    from t2m.examples.tensor_kernels import matrix_add, runtime_prefix
    source = tmp_path / "conflicting.ttir"
    source.write_text(capture_ttir(matrix_add, {"x": "*fp32", "y": "*fp32", "out": "*fp32", "M": "constexpr", "N": "constexpr"}, {"M": 4, "N": 8}))
    spec = TensorABI("matrix_add", (TensorBuffer("x", (4, 8)), TensorBuffer("y", (4, 8)), TensorBuffer("out", (4, 8), "readwrite")), grid=(2,))
    with pytest.raises(ConversionError, match="ownership is not proved"):
        convert_shared(source, spec, tmp_path / "conflicting", optimizer=OPT, memory_profile="buffer")
    assert not (tmp_path / "conflicting/memory-ir").exists()
    from t2m.scalar_abi import RuntimeScalar
    source = tmp_path / "partial.ttir"
    source.write_text(capture_ttir(runtime_prefix, {"x": "*fp32", "n": "i32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    spec = TensorABI("runtime_prefix", (TensorBuffer("x", (13,)), TensorBuffer("out", (13,), "write")), grid=(2,),
                     runtime_scalars=(RuntimeScalar("n", "i32", 1, bounds=(0, 13)),))
    with pytest.raises(ConversionError, match="no proved complete overwrite"):
        convert_shared(source, spec, tmp_path / "partial", optimizer=OPT, memory_profile="buffer")
