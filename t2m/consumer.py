"""Optional checks through an existing Merlin checkout, without changing it."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = '''
import hashlib, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "src"))
from merlin.common.mlir_query import forward_signature
from merlin.llvmlower.c_runtime import generate
capture, out = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
signature = forward_signature(capture / "model.mlir")
generate(capture, out, capture / "inputs.npz", capture / "extra.npz")
files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
         for p in out.iterdir() if p.is_file()}
print(json.dumps({"schema": "t2m.merlin-interface-check.v1", "forward_signature": signature,
                 "parser": "merlin.frontends.linalg_mlir", "runtime": "merlin.llvmlower.c_runtime.generate",
                 "interface_ingestion": "passed", "runtime_generation": "passed",
                 "target_execution": "not-performed", "generated_artifacts": files}))
'''


def check_merlin_interface(capture, work, *, merlin_root, python=None):
    capture, work, root = Path(capture), Path(work), Path(merlin_root).resolve()
    if not (root / "src/merlin/llvmlower/c_runtime.py").is_file():
        raise ValueError("Merlin checkout does not expose the expected model runtime interface")
    work.mkdir(parents=True, exist_ok=False)
    try:
        process = subprocess.run([str(python or sys.executable), "-c", SCRIPT, str(root),
                                  str(capture.resolve()), str((work / "runtime").resolve())],
                                 capture_output=True, text=True, timeout=30,
                                 env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                          TMPDIR=str(work.resolve())))
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Merlin interface qualification exceeded 30 seconds") from exc
    if process.returncode:
        raise ValueError("Merlin interface qualification failed: " + process.stderr[-4000:])
    report = json.loads(process.stdout)
    report["model_sha256"] = hashlib.sha256((capture / "model.mlir").read_bytes()).hexdigest()
    report["merlin_root"] = str(root)
    for name in ("src/merlin/frontends/linalg_mlir.py", "src/merlin/llvmlower/c_runtime.py"):
        report.setdefault("consumer_source_sha256", {})[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (work / "interface.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
