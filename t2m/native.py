"""Bounded native host execution using an existing torch-mlir environment.

This executes serialized standard IR after LLVM legalization. It does not
execute Triton source, use triton-cpu, or establish device/consumer support.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def execute_native(mlir_text, inputs, work, *, python=None, shape_abi=None):
    work = Path(work)
    arrays = [np.asarray(x) for x in inputs]
    if sum(x.nbytes for x in arrays) > 64 * 1024 * 1024:
        raise ValueError("native qualification inputs exceed the 64MiB budget")
    if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1:]):
        raise ValueError("native execution requires independently allocated inputs")
    work.mkdir(parents=True, exist_ok=False)
    (work / "model.mlir").write_text(mlir_text)
    if shape_abi is not None:
        (work / "shape-abi.json").write_text(json.dumps(shape_abi, sort_keys=True) + "\n")
    np.savez(work / "inputs.npz", **{f"in{i}": x for i, x in enumerate(arrays)})
    runner = Path(__file__).with_name("native_runner.py")
    try:
        process = subprocess.run([str(python or sys.executable), str(runner), str(work.resolve())],
                                 capture_output=True, text=True, timeout=30,
                                 env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                          TMPDIR=str(work.resolve())))
    except subprocess.TimeoutExpired as exc:
        raise ValueError("native execution exceeded its 30-second budget") from exc
    if process.returncode:
        raise ValueError("native host execution failed: " + process.stderr[-4000:])
    report = json.loads((work / "execution.json").read_text())
    report["input_sha256"] = hashlib.sha256(mlir_text.encode()).hexdigest()
    with np.load(work / "outputs.npz", allow_pickle=False) as data:
        outputs = tuple(data[f"out{i}"].copy() for i in range(len(data.files)))
    return outputs, report
