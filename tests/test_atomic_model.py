"""Real model2MLIR atomics retain weights, tuple results and actual Merlin execution."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE, M2M = (os.environ.get(n) for n in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON"))
MERLIN, CLANG = os.environ.get("T2M_MERLIN_ROOT"), os.environ.get("T2M_MERLIN_CLANG")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and M2M), reason="existing shared/native/model2MLIR environments required")


@pytest.fixture(scope="module", params=["add", "cas"])
def atomic_model_inputs(tmp_path_factory, request):
    kind = request.param
    tmp_path = tmp_path_factory.mktemp("actual-atomic-model")
    from t2m.examples.atomic_kernels import indexed_atomic, indexed_cas
    capture = tmp_path / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_atomic_model.py"
    process = subprocess.run([M2M, str(script), "--out", str(capture), "--kind", kind], capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr
    constants = {"n": 13, "m": 17, "B": 8, "OP": "add", "SCALAR": False, "MASK": True, "FRESH": False,
                 "SEM": "relaxed", "SCOPE": "gpu"}
    source = tmp_path / "source.ttir"
    if kind == "cas":
        constants = {"B": 8, "SCALAR": False, "SEM": "relaxed", "SCOPE": "gpu"}
        source.write_text(capture_ttir(indexed_cas, {**{n: "*i32" for n in ("state", "comparison", "value", "old")},
            **{name: "constexpr" for name in constants}}, constants))
        abi = TensorABI("indexed_cas", (TensorBuffer("state", (17,), "readwrite", dtype="i32"),
            TensorBuffer("comparison", (16,), dtype="i32"), TensorBuffer("value", (16,), dtype="i32"),
            TensorBuffer("old", (16,), "write", dtype="i32")), grid=(2,))
    else:
        source.write_text(capture_ttir(indexed_atomic, {"x": "*i32", "index": "*i32", "state": "*i32", "old": "*i32",
            **{name: "constexpr" for name in constants}}, constants))
        abi = TensorABI("indexed_atomic", (TensorBuffer("x", (13,), dtype="i32"), TensorBuffer("index", (13,), dtype="i32"),
            TensorBuffer("state", (17,), "readwrite", dtype="i32"), TensorBuffer("old", (13,), "write", dtype="i32")), grid=(2,))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
                            address_policy="source-words", effect_profile="atomics")
    return capture, KernelBinding("t2m_demo::atomic_update", result,
        operands={"state": "state", "comparison": "index", "value": "x"} if kind == "cas" else {}), kind


def test_real_atomic_model_keeps_source_effects_weights_and_tuple_results(tmp_path, atomic_model_inputs):
    capture, binding, kind = atomic_model_inputs
    out = tmp_path / "bound"
    write_model_bundle(capture, [binding], out, work=tmp_path / "qualification", python=NATIVE,
        merlin_root=MERLIN, merlin_python=os.environ.get("T2M_MERLIN_PYTHON"), merlin_host=bool(MERLIN and CLANG), merlin_clang=CLANG)
    receipt = verify_bundle(out)
    assert "kernels/0/memory-admission.json" in receipt["artifacts"]
    admission = json.loads((out / "kernels/0/memory-admission.json").read_text())
    assert admission["schema"] == "t2m.memory-admission.v3"
    assert admission["atomic_adaptations"][0]["operation"] == kind
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    execution = json.loads((out / "execution.json").read_text())
    assert execution["result_count"] == 2 and set(execution["result_reference_members"]) == {"golden.npy", "goldens.npz"}
    differences = json.loads((out / "compatibility.json").read_text())["kernel_profiles"]["kernels/0"]["differences"]
    assert any("private-invocation integer atomics" in item for item in differences)
    if MERLIN:
        assert (out / "consumer/compiler/source/kernels/0/memory-admission.json").read_bytes() == (out / "kernels/0/memory-admission.json").read_bytes()
        if CLANG:
            execution = json.loads((out / "consumer/host/execution.json").read_text())
            assert execution["host_execution"] == execution["reference_comparison"] == "passed"
            assert execution["evaluation"]["functional_inputs_preserved"]
            assert execution["result_count"] == 2
    tampered = tmp_path / "tampered"
    shutil.copytree(out, tampered)
    execution = json.loads((tampered / "execution.json").read_text())
    execution.pop("result_reference_members")
    (tampered / "execution.json").write_text(json.dumps(execution))
    receipt["artifacts"]["execution.json"] = _record(tampered / "execution.json")
    (tampered / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="every result-reference member"):
        verify_bundle(tampered)


@pytest.mark.parametrize("change", ["missing", "shape", "dtype", "unbound"])
def test_all_parent_result_references_must_match_the_complete_model_abi(tmp_path, atomic_model_inputs, change):
    capture, binding, _ = atomic_model_inputs
    broken = tmp_path / "parent"
    shutil.copytree(capture, broken)
    receipt = json.loads((broken / "capture_receipt.json").read_text())
    if change == "unbound":
        receipt["artifacts"].pop("goldens.npz")
    else:
        with np.load(broken / "goldens.npz", allow_pickle=False) as references:
            arrays = {name: references[name].copy() for name in references.files}
        if change == "missing":
            arrays.pop("out1")
        else:
            metadata = json.loads((broken / "meta.json").read_text())
            if change == "shape":
                arrays["out1"] = arrays["out1"][:-1]
                metadata["output_abi"][1]["shape"] = list(arrays["out1"].shape)
            else:
                arrays["out1"] = arrays["out1"].astype(np.float32)
                metadata["result_references"]["storage_dtypes"][1] = "float32"
            (broken / "meta.json").write_text(json.dumps(metadata))
            receipt["artifacts"]["meta.json"] = _record(broken / "meta.json")
        np.savez(broken / "goldens.npz", **arrays)
        receipt["artifacts"]["goldens.npz"] = _record(broken / "goldens.npz")
    (broken / "capture_receipt.json").write_text(json.dumps(receipt))
    out, work = tmp_path / "bound", tmp_path / "qualification"
    with pytest.raises(ValueError):
        write_model_bundle(broken, [binding], out, work=work, python=NATIVE)
    assert not out.exists() and not (work / "native").exists()
