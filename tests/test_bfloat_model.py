"""Real BF16 model captures retain exact source words, learned state and results."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

OPT, NATIVE, M2M = (os.environ.get(n) for n in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON"))
MERLIN, CLANG = os.environ.get("T2M_MERLIN_ROOT"), os.environ.get("T2M_MERLIN_CLANG")
REQUIRED = pytest.mark.skipif(not (OPT and NATIVE and M2M and MERLIN and CLANG), reason="existing shared/model2MLIR/Merlin required")


def translate(root, mode, profile):
    from t2m.examples.bfloat_kernels import bfloat_observe
    from t2m.examples.tensor_kernels import accumulated_matmul
    from t2m.frontend import capture_ttir
    from t2m.model_binding import KernelBinding
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    if mode == "observe":
        kernel, constants, grid = bfloat_observe, {"N": 17, "B": 8}, (3,)
        buffers = (TensorBuffer("x", (17,), dtype="bf16"), TensorBuffer("weight", (17,), dtype="bf16"),
            TensorBuffer("out", (17,), "write", "bf16"), TensorBuffer("words", (17,), "write", "i16"),
            TensorBuffer("value", (17,), "write", "f32"))
        operands = {"x": "x", "weight": "weight"}
    else:
        kernel, constants, grid = accumulated_matmul, {"M": 16, "N": 16, "K": 16}, (1,)
        buffers = (TensorBuffer("x", (16, 16), dtype="bf16"), TensorBuffer("y", (16, 16), dtype="bf16"),
            TensorBuffer("initial", (16, 16)), TensorBuffer("out", (16, 16), "write"))
        operands = {"x": "x", "y": "weight", "initial": "initial"}
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, {**{b.name: "*fp32" if b.dtype == "f32" else "*" + b.dtype for b in buffers},
        **{name: "constexpr" for name in constants}}, constants))
    result = convert_shared(source, TensorABI(kernel.__name__, buffers, grid=grid), root / "translation", optimizer=OPT,
        memory_profile=profile, address_policy="source-words" if profile == "guarded-buffer" else "structured")
    return KernelBinding("t2m_demo::bfloat_" + mode, result, operands=operands)


def capture(root, mode, *, legacy=False):
    out = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_bfloat_model.py"
    process = subprocess.run([M2M, str(script), "--out", str(out), "--mode", mode, *(["--finite"] if legacy else [])], capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert process.returncode == 0, process.stdout + process.stderr
    if legacy:
        # Explicit protocol projection of real source/weights/value archives,
        # exercising the legacy layout without claiming a historical tool run.
        from t2m.bundle import _record
        (out / "tensor_words.npz").unlink()
        meta = json.loads((out / "meta.json").read_text())
        meta.pop("tensor_words")
        (out / "meta.json").write_text(json.dumps(meta))
        receipt = json.loads((out / "capture_receipt.json").read_text())
        receipt["artifacts"].pop("tensor_words.npz")
        receipt["artifacts"]["meta.json"] = _record(out / "meta.json")
        receipt["archive_projection"] = "legacy-value-archives; original core bytes unchanged"
        (out / "capture_receipt.json").write_text(json.dumps(receipt))
    return out


@REQUIRED
@pytest.mark.parametrize("mode,profile,legacy", [(mode, profile, False) for mode in ("observe", "matmul")
    for profile in ("tensor", "buffer", "guarded-buffer")] + [(mode, "guarded-buffer", True) for mode in ("observe", "matmul")])
def test_model_exact_bfloat_words_weights_and_all_references(tmp_path, mode, profile, legacy):
    from t2m.bundle import verify_bundle
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_words import TensorWords
    parent = capture(tmp_path, mode, legacy=legacy)
    binding = translate(tmp_path, mode, profile)
    out, work = tmp_path / "model", tmp_path / "qualification"
    write_model_bundle(parent, [binding], out, work=work, python=NATIVE, merlin_root=MERLIN,
        merlin_python=os.environ.get("T2M_MERLIN_PYTHON"), merlin_host=True, merlin_clang=CLANG)
    receipt = verify_bundle(out)
    assert receipt["materialized_abi"] == "unchanged-from-parent"
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "extra.npz", "golden.npy", *(() if legacy else ("tensor_words.npz",))):
        assert (out / name).read_bytes() == (parent / name).read_bytes()
    execution = json.loads((out / "execution.json").read_text())
    assert execution["result_count"] == (6 if mode == "observe" else 1)
    assert any(s.get("word_witness") for s in execution["materialized_arguments"])
    assert any(s.get("captured_constant", {}).get("source_dtype") == "bf16" for s in execution["materialized_arguments"])
    stage = json.loads((out / "consumer/compiler/compile-inputs.json").read_text())
    assert "tensor_words.npz" in stage["excluded_evaluator_members"]
    staged_bytes = b"".join((out / "consumer/compiler" / name).read_bytes() for name in stage["members"])
    parent_receipt = json.loads((parent / "capture_receipt.json").read_text())
    if not legacy:
        assert parent_receipt["artifacts"]["tensor_words.npz"]["sha256"].encode() not in staged_bytes
    if mode == "observe":
        assert (out / "goldens.npz").read_bytes() == (parent / "goldens.npz").read_bytes()
        assert ("tensor_words.npz" in execution["result_reference_members"]) != legacy
        with TensorWords(parent) as words, np.load(parent / "goldens.npz", allow_pickle=False) as legacy:
            references = [words.get("goldens.npz", f"out{i}", legacy[f"out{i}"], "bf16" if i in (0, 3) else "i16" if i in (1, 4) else "f32", (17,)) for i in range(6)]
        for path in (work / "native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                for i in (0, 1, 3, 4):
                    np.testing.assert_array_equal(actual[f"out{i}"], references[i])
    else:
        assert "linalg.matmul" in (out / "model.mlir").read_text()
    host = json.loads((out / "consumer/host/execution.json").read_text())
    assert host["host_execution"] == host["reference_comparison"] == "passed"
    assert host["evaluation"]["functional_inputs_preserved"]


@REQUIRED
def test_rehashed_nan_input_word_still_fails_the_independent_bit_reference(tmp_path):
    from t2m.bundle import _record
    from t2m.model_bundle import write_model_bundle
    parent = capture(tmp_path, "observe")
    metadata = json.loads((parent / "meta.json").read_text())
    entry = next(e for e in metadata["tensor_words"]["entries"] if e["archive"] == "inputs.npz" and e["key"] == "in0")
    with np.load(parent / "tensor_words.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays[entry["word_key"]][4] ^= np.uint16(2)  # A different NaN; legacy numeric classification still agrees.
    np.savez(parent / "tensor_words.npz", **arrays)
    receipt = json.loads((parent / "capture_receipt.json").read_text())
    receipt["artifacts"]["tensor_words.npz"] = _record(parent / "tensor_words.npz")
    (parent / "capture_receipt.json").write_text(json.dumps(receipt))
    binding = translate(tmp_path, "observe", "guarded-buffer")
    out = tmp_path / "model"
    with pytest.raises(AssertionError):
        write_model_bundle(parent, [binding], out, work=tmp_path / "qualification", python=NATIVE, merlin_root=MERLIN)
    assert not (out / "capture_receipt.json").exists()
