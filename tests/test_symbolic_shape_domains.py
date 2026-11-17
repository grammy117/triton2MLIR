"""Real dynamic reductions execute source ranges and affine input relations."""

import os
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest

REQUIRED = all(os.environ.get(name) for name in ("T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON"))
pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing model2MLIR and native environments required")


@pytest.mark.parametrize("mode", ["shared", "affine", "inverted", "zero", "unbounded"])
def test_compiled_source_domains_accept_independent_shapes_and_abort_invalid_relations(tmp_path, mode):
    from t2m.native import execute_native, observe_native
    from t2m.shared_tensor import parse_shared

    capture = tmp_path / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_shape_domain_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture), "--mode", mode],
        capture_output=True, text=True, check=False, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stdout + process.stderr
    source = (capture / "model.mlir").read_text()
    parse_shared(source).verify()
    contract = {"schema": "t2m.runtime-shapes.v2", "inputs": [
        {"shape": ["rows_x", 3], "dtype": "f32"}, {"shape": ["rows_y", 3], "dtype": "f32"}],
        "outputs": [{"shape": [3], "dtype": "f32"}],
        "dimension_bindings": {"rows_x": {"argument": 0, "axis": 0}, "rows_y": {"argument": 1, "axis": 0}}}
    counts = (0, 1, 31) if mode == "zero" else (2, 7, 32) if mode == "unbounded" else (2, 7, 31)
    for count in counts:
        a, b = (2 * count + 1 if mode == "inverted" else count), (count if mode == "shared" else 2 * count + 1)
        x = np.arange(a * 3, dtype=np.float32).reshape(a, 3) * .125
        y = np.arange(b * 3, dtype=np.float32).reshape(b, 3) * -.25
        (actual,), report = execute_native(source, [x, y], tmp_path / f"valid-{count}",
                                           python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=contract)
        assert report["llvm_lowering_verified"]
        np.testing.assert_array_equal(actual, x.sum(axis=0) + y.sum(axis=0))
    invalid = [(1, 1 if mode == "shared" else 3), (32, 32 if mode == "shared" else 65), (4, 5)]
    if mode == "inverted":
        invalid = [(3, 3), (65, 65), (8, 8)]
    elif mode == "zero":
        invalid = [(32, 65), (0, 2), (1, 4)]
    elif mode == "unbounded":
        invalid = [(1, 3), (4, 10), (0, 1)]
    for index, (a, b) in enumerate(invalid):
        work = tmp_path / f"invalid-{index}"
        report = observe_native(source, [np.zeros((a, 3), np.float32), np.zeros((b, 3), np.float32)], work,
                                python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=contract)
        assert report["returncode"] == -signal.SIGABRT
        assert b"exported shape domain:" in (work / "stdout.bin").read_bytes() + (work / "stderr.bin").read_bytes()
        assert not (work / "outputs.npz").exists() and not (work / "execution.json").exists()
