"""Optional checks through an existing Merlin checkout, without changing it."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import signal

import numpy as np

SCRIPT = '''
import hashlib, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "src"))
from merlin.common.mlir_query import forward_signature
from merlin.llvmlower.c_runtime import generate
from merlin.frontends.compile_inputs import stage_compile_inputs
capture, out = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
signature = forward_signature(capture / "model.mlir")
stage = stage_compile_inputs(capture, out.parent / "compiler")
generate(capture, out, capture / "inputs.npz", capture / "extra.npz")
files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
         for p in out.iterdir() if p.is_file()}
print(json.dumps({"schema": "t2m.merlin-interface-check.v2", "forward_signature": signature,
                 "parser": "merlin.frontends.linalg_mlir", "runtime": "merlin.llvmlower.c_runtime.generate",
                 "interface_ingestion": "passed", "runtime_generation": "passed",
                 "compiler_input_staging": "passed", "compiler_input_manifest": stage,
                 "target_execution": "not-performed", "generated_artifacts": files}))
'''


def check_merlin_interface(capture, work, *, merlin_root, python=None):
    capture, work, root = Path(capture), Path(work), Path(merlin_root).resolve()
    if not (root / "src/merlin/llvmlower/c_runtime.py").is_file():
        raise ValueError("Merlin checkout does not expose the expected model runtime interface")
    work.mkdir(parents=True, exist_ok=False)
    (work / "input-receipt.json").write_bytes((capture / "capture_receipt.json").read_bytes())
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
    for name in ("src/merlin/frontends/linalg_mlir.py", "src/merlin/llvmlower/c_runtime.py",
                 "src/merlin/frontends/compile_inputs.py", "src/merlin/frontends/derived_compile_inputs.py",
                 "src/merlin/xdsl_dialects/standard_tensor.py"):
        report.setdefault("consumer_source_sha256", {})[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (work / "interface.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def check_merlin_host(work, arguments, *, merlin_root, python=None, compiler_python=None, clang=None):
    """Compile only staged inputs, then invoke the actual Merlin host library."""
    work, root = Path(work), Path(merlin_root).resolve()
    host = work / "host"
    host.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MERLIN_COMPILE_TIMEOUT_S="30", TMPDIR=str(host.resolve()))
    if compiler_python is not None:
        environment["MERLIN_COMPILER_PYTHON"] = str(compiler_python)
    if clang is not None:
        environment["MERLIN_CLANG"] = str(clang)
    for mode, extra in (("compile", []), ("execute", [str(Path(arguments).resolve())])):
        _host_worker(mode, root, work / "compiler", host, extra, environment, python)
    report = json.loads((host / "compilation.json").read_text())
    report["evaluation"] = json.loads((host / "evaluation.json").read_text())
    with np.load(host / "outputs.npz", allow_pickle=False) as data:
        outputs = tuple(data[f"out{i}"].copy() for i in range(len(data.files)))
    return outputs, report


def _host_worker(mode, root, staged, host, extra, environment, python):
    worker = str(Path(__file__).with_name("consumer_worker.py"))
    command = [str(python or sys.executable), worker, mode, str(root),
               str(Path(staged).resolve()), str(Path(host).resolve()), *extra]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=environment, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise ValueError("Merlin host " + mode + " exceeded 60 seconds") from exc
    if process.returncode:
        raise ValueError("Merlin host " + mode + " failed: " + (stdout + stderr)[-4000:])


def invoke_merlin_host(work, arguments, out, *, merlin_root, python=None):
    """Invoke an existing compiled library; there is no compiler call here."""
    work, out, root = Path(work), Path(out).resolve(), Path(merlin_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(out))
    _host_worker("invoke", root, work / "compiler", work / "host",
                 [str(Path(arguments).resolve()), str(out)], environment, python)
    report = json.loads((out / "evaluation.json").read_text())
    with np.load(out / "outputs.npz", allow_pickle=False) as values:
        outputs = tuple(values[f"out{i}"].copy() for i in range(len(values.files)))
    return outputs, report
