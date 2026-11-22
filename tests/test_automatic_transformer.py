"""Real launch discovery and complete two-block public Merlin execution."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from t2m.automatic_model import translate_launches
from t2m.shared_tensor import parse_shared

REPO = Path(__file__).resolve().parents[1]
TORCH = os.environ.get("T2M_MODEL2MLIR_PYTHON")
TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


@pytest.mark.skipif(not TORCH, reason="existing model2MLIR Torch environment required")
@pytest.mark.parametrize("mode", ("positive", "wrapper-math", "two-launches"))
def test_actual_launch_graph_controls_bindings_and_rejects_omitted_work(tmp_path, mode):
    process = subprocess.run([TORCH, str(REPO / "tests/torch_discovery_worker.py"), mode],
        text=True, capture_output=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stdout + process.stderr
    record = json.loads(process.stdout.splitlines()[-1])
    if mode == "positive":
        call = record["calls"][0]
        assert record["tensor_values_used"] is False and call["buffers"][0]["shape"] == [13, 16]
        assert call["results"] == ["out"] and len(call["operands"]) == 11
        call["source"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="source changed"):
            translate_launches(record, tmp_path / "changed-source")
        assert not list((tmp_path / "changed-source").rglob("*.mlir"))
    else:
        assert "wrapper contains computation" in record["refused"] if mode == "wrapper-math" else "one Triton launch" in record["refused"]


@pytest.mark.skipif(not (TOOLS and TORCH), reason="existing native/model/Merlin tools required")
def test_ragged_transformer_blocks_automatically_bind_and_execute_all_results(tmp_path):
    root = tmp_path / "demo"
    process = subprocess.run([sys.executable, str(REPO / "scripts/run_transformer_demo.py"),
        "--out", str(root), "--tokens", "13"], text=True, capture_output=True, timeout=300, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    (tmp_path / "demo.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout[-3000:] + process.stderr[-6000:]
    report = json.loads((root / "summary.json").read_text())
    assert report["native_reference_comparison"] == report["public_merlin_host_execution"] == "passed"
    assert report["manual_kernel_bindings"] is report["gpu_kernel_executed"] is False
    assert report["artifact_only_publication"] is True
    assert (root / "published/host/model.so").read_bytes() == (root / "compiled/host/model.so").read_bytes()
    assert not any(path.name in {"inputs.npz", "golden.npy", "goldens.npz", "extra.npz"}
                   for path in (root / "published").rglob("*"))
    assert len(report["invocations"]) == 3 and all(row["results"] == 2 for row in report["invocations"])
    trace = json.loads((root / "bound/kernel-bindings.json").read_text())
    assert len(trace["calls"]) == len(trace["kernels"]) == 1 and trace["model_abi"] == "unchanged"
    kernel = next(iter(trace["kernels"].values()))
    assert sum(row["opcode"] == "tt.dot" for row in kernel["source_operations"]) == 8
    assert kernel["automatic_launch"]["tensor_values_used"] is False
    before = parse_shared((root / "capture/parent/model.mlir").read_text())
    assert any(op.attributes.get("prov.fqn") is not None and op.attributes["prov.fqn"].data == "regular"
               and op.name.startswith("linalg.") for op in before.walk())
    signature = json.loads((root / "compiled/compiler/signature.json").read_text())
    assert signature["inputs"] == [{"dtype": "f32", "shape": [13, 16]}]
    bound = parse_shared((root / "bound/model.mlir").read_text())
    assert not any(op.name == "func.call" for op in bound.walk())
