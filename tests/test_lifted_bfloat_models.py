"""Real source literal arguments reach Merlin without evaluator archive access."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_bfloat_model import REQUIRED, M2M, NATIVE, MERLIN, CLANG, translate


@REQUIRED
@pytest.mark.parametrize("mode", ["observe", "matmul"])
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_actual_lifted_source_constants_keep_types_bytes_and_every_result(tmp_path, mode, profile):
    from t2m.bundle import verify_bundle
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_words import TensorWords
    source = tmp_path / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_bfloat_model.py"
    captured = subprocess.run([M2M, str(script), "--out", str(source), "--mode", mode, "--lifted"], capture_output=True,
        text=True, timeout=60, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert captured.returncode == 0, captured.stdout + captured.stderr
    parent = json.loads((source / "capture_receipt.json").read_text())
    assert len(parent["compiler_constants"]["bindings"]) == 2
    binding = translate(tmp_path, mode, profile)
    out, work = tmp_path / "model", tmp_path / "qualification"
    write_model_bundle(source, [binding], out, work=work, python=NATIVE, merlin_root=MERLIN,
        merlin_python=os.environ.get("T2M_MERLIN_PYTHON"), merlin_host=True, merlin_clang=CLANG)
    verify_bundle(out)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "extra.npz", "golden.npy", "tensor_words.npz"):
        assert (out / name).read_bytes() == (source / name).read_bytes()
    stage = json.loads((out / "consumer/compiler/compile-inputs.json").read_text())
    assert "compiler-constants.json" in stage["members"]
    projection = json.loads((out / "consumer/compiler/compiler-constants.json").read_text())
    assert projection["bindings"] == parent["compiler_constants"]["bindings"]
    staged = b"".join((out / "consumer/compiler" / name).read_bytes() for name in stage["members"])
    for name in ("extra.npz", "inputs.npz", "golden.npy", "tensor_words.npz"):
        assert parent["artifacts"][name]["sha256"].encode() not in staged
    execution = json.loads((out / "execution.json").read_text())
    literals = [item for item in execution["materialized_arguments"] if item.get("captured_constant", {}).get("manifest_kind") == "constant"]
    assert len(literals) == (2 if mode == "observe" else 1)  # BF16 uses the exact byte reader; F32 uses safetensors directly.
    host = json.loads((out / "consumer/host/execution.json").read_text())
    assert host["host_execution"] == host["reference_comparison"] == "passed"
    if mode == "observe":
        with TensorWords(source) as words, np.load(source / "goldens.npz", allow_pickle=False) as legacy:
            reference = words.get("goldens.npz", "out0", legacy["out0"], "bf16", [17])
            with np.load(out / "consumer/host/outputs.npz", allow_pickle=False) as actual:
                np.testing.assert_array_equal(actual["out0"], reference)
                for i in (1, 4):
                    np.testing.assert_array_equal(actual[f"out{i}"], legacy[f"out{i}"])
    else:
        assert "linalg.matmul" in (out / "model.mlir").read_text()
