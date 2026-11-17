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
stage = stage_compile_inputs(capture, out.parent / "compiler", result_mode=sys.argv[5] or None)
staged_signature = json.loads((out.parent / "compiler/signature.json").read_text())
execution_owned = staged_signature["schema"] == "merlin.compile-signature.v6"
if execution_owned:
    pass  # Result descriptors are produced by the separate host invocation.
elif staged_signature["schema"] in ("merlin.compile-signature.v3", "merlin.compile-signature.v4", "merlin.compile-signature.v5"):
    # The caller supplies the separately materialized full positional roster.
    # Its values size this runtime harness and never enter compiler staging.
    if not sys.argv[4]:
        raise ValueError("symbolic runtime generation requires positional invocation arguments")
    arguments = pathlib.Path(sys.argv[4])
    generate(capture, out, arguments, capture / "extra.npz",
             runtime_signature=out.parent / "compiler/signature.json")
else:
    generate(capture, out, capture / "inputs.npz", capture / "extra.npz")
files = {} if execution_owned else {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
         for p in out.iterdir() if p.is_file()}
print(json.dumps({"schema": "t2m.merlin-interface-check.v3" if execution_owned else "t2m.merlin-interface-check.v2", "forward_signature": signature,
                 "parser": "merlin.frontends.linalg_mlir", "runtime": "merlin.llvmlower.owned_results" if execution_owned else "merlin.llvmlower.c_runtime.generate",
                 "interface_ingestion": "passed", "runtime_generation": "not-performed; execution-produced result descriptors" if execution_owned else "passed",
                 "compiler_input_staging": "passed", "compiler_input_manifest": stage,
                 "target_execution": "not-performed", "generated_artifacts": files}))
'''


def check_merlin_interface(capture, work, *, merlin_root, python=None, arguments=None, result_mode=None):
    capture, work, root = Path(capture), Path(work), Path(merlin_root).resolve()
    if not (root / "src/merlin/llvmlower/c_runtime.py").is_file():
        raise ValueError("Merlin checkout does not expose the expected model runtime interface")
    work.mkdir(parents=True, exist_ok=False)
    (work / "input-receipt.json").write_bytes((capture / "capture_receipt.json").read_bytes())
    try:
        process = subprocess.run([str(python or sys.executable), "-c", SCRIPT, str(root),
                                  str(capture.resolve()), str((work / "runtime").resolve()),
                                  str(Path(arguments).resolve()) if arguments is not None and result_mode is None else "",
                                  result_mode or ""],
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
                 "src/merlin/xdsl_dialects/standard_tensor.py", "src/merlin/frontends/derived_runtime_arguments.py",
                 "src/merlin/frontends/derived_live_arguments.py", "src/merlin/frontends/source_index_ranges.py",
                 "src/merlin/frontends/model_shape_facts.py", "src/merlin/frontends/derived_symbolic_shapes.py",
                 "src/merlin/frontends/shape_expressions.py", "src/merlin/frontends/shape_expression_ir.py",
                 "src/merlin/frontends/shape_input_words.py", "src/merlin/frontends/shape_word_bindings.py",
                 "src/merlin/llvmlower/runtime_invocation.py"):
        if (root / name).is_file():
            report.setdefault("consumer_source_sha256", {})[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if result_mode is not None:
        for name in ("src/merlin/frontends/execution_result_signature.py", "src/merlin/llvmlower/owned_result_program.py",
                     "src/merlin/frontends/execution_model_correspondence.py", "src/merlin/frontends/execution_call_contracts.py",
                     "src/merlin/llvmlower/owned_results.py", "src/merlin/llvmlower/owned_result_abi.py",
                     "src/merlin/llvmlower/owned_result_allocations.py", "src/merlin/llvmlower/execution_result_inputs.py",
                     "merlin/runtime/abi/owned_results.c", "src/merlin/llvmlower/owned_result_c.py",
                     "merlin/runtime/abi/owned_result_views.c", "merlin/runtime/abi/owned_result_views.h"):
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


def observe_merlin_host(work, arguments, out, *, merlin_root, python=None):
    from .process_outcome import observe
    work, out = Path(work).resolve(), Path(out).absolute()
    out.mkdir(parents=True,exist_ok=False)
    command = [str(python or sys.executable),str(Path(__file__).with_name("consumer_worker.py")),"observe",
        str(Path(merlin_root).resolve()),str(work / "compiler"),str(work / "host"),str(Path(arguments).resolve()),str(out)]
    return observe(command,out,timeout=60,environment=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TMPDIR=str(out)))


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
