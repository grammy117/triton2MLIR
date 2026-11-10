"""Independent Merlin qualification of a complete standalone tensor package."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .bundle import _json, _record, verify_bundle
from .consumer import check_merlin_host
from .tensor_storage import assert_tensor_close


STAGE = '''
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "src"))
from merlin.frontends.compile_inputs import stage_compile_inputs
stage_compile_inputs(sys.argv[2], sys.argv[3])
'''


def qualify_kernel(capture, out, *, merlin_root, python=None, compiler_python=None, clang=None, rtol=0, atol=0):
    """Keep the source package immutable; publish success only after real execution."""
    capture, out, root = Path(capture).resolve(), Path(out).absolute(), Path(merlin_root).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if not np.isfinite(rtol) or not np.isfinite(atol) or min(rtol, atol) < 0:
        raise ValueError("Merlin kernel comparison needs finite nonnegative tolerances")
    receipt = verify_bundle(capture)
    if receipt["schema"] == "t2m.effect-kernel-receipt.v1":
        if rtol or atol:
            raise ValueError("effect references use exact outcomes, not numeric tolerances")
        from .effect_consumer import qualify_effect_kernel
        return qualify_effect_kernel(capture,out,merlin_root=root,python=python,compiler_python=compiler_python,clang=clang)
    if receipt["schema"] != "t2m.capture-receipt.v2":
        raise ValueError("standalone Merlin qualification requires the shared functional tensor profile")
    if capture in out.parents:
        raise ValueError("qualification output must be outside the source capture")
    out.mkdir(parents=True, exist_ok=False)
    (out / "input-receipt.json").write_bytes((capture / "capture_receipt.json").read_bytes())
    try:
        process = subprocess.run([str(python or sys.executable), "-c", STAGE, str(root), str(capture), str(out / "compiler")],
            capture_output=True, text=True, timeout=30,
            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(out)))
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Merlin standalone compiler staging exceeded 30 seconds") from exc
    if process.returncode:
        raise ValueError("Merlin standalone staging failed: " + process.stderr[-4000:])
    outputs, report = check_merlin_host(out, capture / "inputs.npz", merlin_root=root, python=python,
        compiler_python=compiler_python, clang=clang)
    references = [np.load(capture / "golden.npy", allow_pickle=False)]
    with np.load(capture / "extra.npz", allow_pickle=False) as extra:
        references.extend(extra[f"out{i}"].copy() for i in range(1, len(outputs)))
    if len(outputs) != len(references):
        raise ValueError("Merlin kernel result count differs from independent references")
    types = json.loads((out / "compiler/signature.json").read_text())["outputs"]
    for actual, golden, type_ in zip(outputs, references, types, strict=True):
        if actual.shape != golden.shape or actual.dtype != golden.dtype:
            raise ValueError("Merlin kernel result shape/dtype differs from independent reference")
        assert_tensor_close(actual, golden, type_["dtype"], rtol=rtol, atol=atol)
    # Detect a source-package change during staging or invocation. The compiler
    # never receives these evaluator values or the qualification report.
    if verify_bundle(capture) != receipt or _record(capture / "capture_receipt.json") != _record(out / "input-receipt.json"):
        raise ValueError("source capture changed during Merlin qualification")
    report.update(schema="t2m.merlin-kernel-host-qualification.v1", host_execution="passed",
                  reference_comparison="passed", reference_artifacts={name: _record(capture / name)
                      for name in ("golden.npy", "extra.npz")}, rtol=rtol, atol=atol,
                  source_kernel_execution="not-performed", accelerator_execution="not-performed")
    _json(out / "host/execution.json", report)
    qualified = {"schema": "t2m.merlin-kernel-qualification.v1", "source_directory": str(capture),
                 "source_receipt": _record(out / "input-receipt.json"),
                 "artifacts": {str(p.relative_to(out)): _record(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    _json(out / "qualification.json", qualified)
    try:
        verify_kernel_qualification(out)
    except BaseException:
        (out / "qualification.json").unlink(missing_ok=True)
        raise
    return qualified


def verify_kernel_qualification(out):
    """Check retained consumer/source/library/reference relationships and bytes."""
    out = Path(out)
    receipt = json.loads((out / "qualification.json").read_text())
    if receipt.get("schema") == "t2m.merlin-effect-qualification.v1":
        from .effect_consumer import verify_effect_qualification
        return verify_effect_qualification(out)
    required = {"input-receipt.json", "compiler/compile-inputs.json", "compiler/program.mlir", "compiler/signature.json",
                "host/model.so", "host/outputs.npz", "host/execution.json", "host/evaluation.json", "host/compilation.json"}
    if receipt.get("schema") != "t2m.merlin-kernel-qualification.v1" or not required <= set(receipt.get("artifacts", {})):
        raise ValueError("incomplete Merlin kernel qualification")
    for name, expected in receipt["artifacts"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or name == "qualification.json":
            raise ValueError("invalid qualification artifact path")
        path = out / relative
        if (any(out.joinpath(*relative.parts[:i]).is_symlink() for i in range(len(relative.parts) + 1))
                or not path.is_file() or _record(path) != expected):
            raise ValueError("Merlin qualification artifact integrity mismatch: " + name)
    source = json.loads((out / "input-receipt.json").read_text())
    stage = json.loads((out / "compiler/compile-inputs.json").read_text())
    report = json.loads((out / "host/execution.json").read_text())
    if (source.get("schema") != "t2m.capture-receipt.v2"
            or receipt["source_receipt"] != _record(out / "input-receipt.json")
            or stage.get("source_profile") != "standalone-tensor-kernel"
            or stage["source"]["capture_receipt_sha256"] != receipt["source_receipt"]["sha256"]
            or stage["source"]["model_mlir_sha256"] != source["artifacts"]["model.mlir"]["sha256"]):
        raise ValueError("Merlin kernel compiler stage does not bind original source receipt")
    for name, digest in stage["members"].items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or "compiler/" + name not in receipt["artifacts"]
                or _record(out / "compiler" / relative)["sha256"] != digest):
            raise ValueError("Merlin compiler stage member differs from manifest")
    if (report.get("schema") != "t2m.merlin-kernel-host-qualification.v1"
            or report.get("host_execution") != "passed" or report.get("reference_comparison") != "passed"
            or report.get("source_program") != _record(out / "compiler/program.mlir")
            or report.get("shared_library") != _record(out / "host/model.so")
            or report.get("evaluation", {}).get("outputs") != _record(out / "host/outputs.npz")
            or report.get("evaluation", {}).get("functional_inputs_preserved") is not True
            or report.get("reference_artifacts") != {name: source["artifacts"][name] for name in ("golden.npy", "extra.npz")}):
        raise ValueError("Merlin kernel host qualification has inconsistent source/result evidence")
    return receipt
