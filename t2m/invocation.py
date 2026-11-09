"""Repeated evaluator-only invocation of a qualified Merlin native library."""

import json
from pathlib import Path
import shutil

import numpy as np

from .bundle import _json, _record
from .consumer import invoke_merlin_host
from .kernel_consumer import verify_kernel_qualification


def invoke_kernel(qualification, arguments, reference, out, *, merlin_root, python=None, rtol=0, atol=0):
    qualification, arguments, reference = Path(qualification), Path(arguments), Path(reference)
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if not np.isfinite(rtol) or not np.isfinite(atol) or min(rtol, atol) < 0:
        raise ValueError("invocation comparison needs finite nonnegative tolerances")
    verified = verify_kernel_qualification(qualification)
    if qualification.resolve() in out.parents:
        raise ValueError("invocation evidence must be outside its immutable qualification")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("qualification.json", "compiler/signature.json", "compiler/compile-inputs.json", "host/compilation.json"):
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(qualification / name, target)
    for name, source in (("inputs.npz", arguments), ("reference.npz", reference)):
        target = out / "evaluator" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    outputs, evaluation = invoke_merlin_host(qualification, out / "evaluator/inputs.npz", out / "evaluation",
                                            merlin_root=merlin_root, python=python)
    with np.load(out / "evaluator/reference.npz", allow_pickle=False) as expected:
        if set(expected.files) != {f"out{i}" for i in range(len(outputs))}:
            raise ValueError("invocation reference needs exactly the declared result keys")
        for i, actual in enumerate(outputs):
            golden = expected[f"out{i}"]
            if actual.shape != golden.shape or actual.dtype != golden.dtype:
                raise ValueError("invocation reference shape/dtype differs from actual result")
            np.testing.assert_allclose(actual, golden, rtol=rtol, atol=atol, equal_nan=True)
    if verify_kernel_qualification(qualification) != verified or _record(qualification / "qualification.json") != _record(out / "source/qualification.json"):
        raise ValueError("compiled qualification changed during invocation")
    report = {"schema": "t2m.merlin-kernel-invocation.v1", "native_compilation": "reused-qualified-library",
              "reference_comparison": "passed", "accelerator_execution": "not-performed",
              "evaluation": evaluation, "rtol": rtol, "atol": atol,
              "artifacts": {str(p.relative_to(out)): _record(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    _json(out / "invocation.json", report)
    try:
        verify_invocation(out)
    except BaseException:
        (out / "invocation.json").unlink(missing_ok=True)
        raise
    return report


def verify_invocation(out):
    out = Path(out)
    report = json.loads((out / "invocation.json").read_text())
    required = {"source/qualification.json", "source/compiler/signature.json", "source/compiler/compile-inputs.json",
                "source/host/compilation.json", "evaluator/inputs.npz", "evaluator/reference.npz",
                "evaluation/outputs.npz", "evaluation/evaluation.json"}
    if (report.get("schema") != "t2m.merlin-kernel-invocation.v1" or report.get("reference_comparison") != "passed"
            or report.get("native_compilation") != "reused-qualified-library" or not required <= set(report.get("artifacts", {}))):
        raise ValueError("not a complete reused-library invocation receipt")
    for name, record in report["artifacts"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or name == "invocation.json":
            raise ValueError("invalid invocation artifact path")
        path = out / relative
        if (any(out.joinpath(*relative.parts[:i]).is_symlink() for i in range(len(relative.parts) + 1))
                or not path.is_file() or _record(path) != record):
            raise ValueError("invocation artifact integrity mismatch: " + name)
    source = json.loads((out / "source/qualification.json").read_text())
    compilation = json.loads((out / "source/host/compilation.json").read_text())
    stage = json.loads((out / "source/compiler/compile-inputs.json").read_text())
    evaluation = json.loads((out / "evaluation/evaluation.json").read_text())
    for original, copied in (("compiler/signature.json", "source/compiler/signature.json"),
                             ("compiler/compile-inputs.json", "source/compiler/compile-inputs.json"),
                             ("host/compilation.json", "source/host/compilation.json")):
        if source["artifacts"].get(original) != _record(out / copied):
            raise ValueError("invocation source snapshot differs from qualified members")
    if (evaluation != report.get("evaluation") or evaluation.get("invocations") != 2
            or evaluation.get("functional_inputs_preserved") is not True
            or evaluation.get("arguments") != _record(out / "evaluator/inputs.npz")
            or evaluation.get("outputs") != _record(out / "evaluation/outputs.npz")
            or evaluation.get("shared_library") != source["artifacts"]["host/model.so"]
            or evaluation.get("shared_library") != compilation.get("shared_library")
            or evaluation.get("source_program", {}).get("sha256") != stage["members"]["program.mlir"]
            or evaluation.get("source_program") != compilation.get("source_program")):
        raise ValueError("invocation does not bind its qualified library/program/arguments/outputs")
    return report
