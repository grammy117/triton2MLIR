"""Compile declared source effects, then independently observe a real host call."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .bundle import _json, _record
from .consumer import _host_worker, observe_merlin_host
from .effect_bundle import SCHEMA, verify_effect_bundle
from .effect_reference import compare_outcome, validate_reference, validate_inputs
from .effect_results import ARCHIVE,reference_members,load_archive,validate_results
from .kernel_consumer import STAGE
from .shared_tensor import TensorABI, parse_shared

QUALIFICATION = "t2m.merlin-effect-qualification.v1"


def _compare(compiler,host,evaluation,evaluator,reference,spec,signature):
    with np.load(evaluator / "inputs.npz",allow_pickle=False) as saved:
        if set(saved.files) != {f"in{i}" for i in range(len(spec.inputs))}:
            raise ValueError("effect comparison requires the complete argument roster")
        inputs = validate_inputs(spec,[saved[f"in{i}"] for i in range(len(spec.inputs))])
    return compare_outcome(evaluation,reference,backend="merlin-host",program=_record(compiler / "program.mlir"),
        arguments=_record(evaluator / "inputs.npz"),implementation={"kind":"shared-library","artifact":_record(host / "model.so")},
        result_count=len(signature["outputs"]),spec=spec,inputs=inputs,numeric_reference=evaluator / ARCHIVE)


def checked_signature(compiler):
    compiler = Path(compiler)
    spec = TensorABI.from_dict(json.loads((compiler / "source/spec.json").read_text()))
    signature = json.loads((compiler / "signature.json").read_text())
    module = parse_shared((compiler / "program.mlir").read_text())
    functions = list(module.ops)
    if (len(functions) != 1 or functions[0].sym_name.data != "forward"
            or tuple(functions[0].function_type.inputs) != tuple(value.type for value in spec.inputs)
            or tuple(functions[0].function_type.outputs) != tuple(value.type for value in spec.outputs)
            or len(signature.get("inputs",[])) != len(spec.inputs) or len(signature.get("outputs",[])) != len(spec.outputs)
            or signature.get("effect_abi") != "source-encoded-observable-effects; evaluator-outcome-excluded"):
        raise ValueError("effect invocation signature differs from its actual source/payload types")
    for declared,value in zip([*signature["inputs"],*signature["outputs"]],[*spec.inputs,*spec.outputs],strict=True):
        scalar = value in spec.runtime_scalars
        if (declared.get("shape") != list(value.shape) or declared.get("dtype") != value.dtype
                or (declared.get("kind") == "scalar") != scalar
                or declared.get("bounds") != (list(value.bounds) if scalar and value.bounds is not None else None)):
            raise ValueError("effect invocation declaration differs from the original typed ABI")
    return spec,signature


def qualify_effect_kernel(capture,out,*,merlin_root,python=None,compiler_python=None,clang=None):
    capture,out,root = Path(capture).resolve(),Path(out).absolute(),Path(merlin_root).resolve()
    source = verify_effect_bundle(capture)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if capture in out.parents:
        raise ValueError("effect qualification must be outside its immutable source capture")
    out.mkdir(parents=True)
    shutil.copyfile(capture / "capture_receipt.json",out / "input-receipt.json")
    process = subprocess.run([str(python or sys.executable),"-c",STAGE,str(root),str(capture),str(out / "compiler")],
        capture_output=True,text=True,timeout=30,env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TMPDIR=str(out)))
    if process.returncode:
        raise ValueError("Merlin effect staging failed: "+process.stderr[-4000:])
    host = out / "host"
    host.mkdir()
    environment = dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MERLIN_COMPILE_TIMEOUT_S="30",TMPDIR=str(host))
    if compiler_python is not None:
        environment["MERLIN_COMPILER_PYTHON"] = str(compiler_python)
    if clang is not None:
        environment["MERLIN_CLANG"] = str(clang)
    _host_worker("compile",root,out / "compiler",host,[],environment,python)
    evaluator = out / "evaluator"
    evaluator.mkdir()
    reference = json.loads((capture / "reference.json").read_text())
    for name in ("inputs.npz",*reference_members(reference)):
        shutil.copyfile(capture / name,evaluator / name)
    observe_merlin_host(out,evaluator / "inputs.npz",out / "evaluation",merlin_root=root,python=python)
    spec,signature = checked_signature(out / "compiler")
    comparison = _compare(out / "compiler",host,out / "evaluation",evaluator,reference,spec,signature)
    compilation = json.loads((host / "compilation.json").read_text())
    _json(host / "execution.json",{**compilation,"schema":"t2m.merlin-effect-host-qualification.v1",**comparison,
        "host_execution":"matched-expected-outcome","reference":_record(evaluator / "reference.json"),
        "observation":_record(out / "evaluation/outcome.json")})
    if verify_effect_bundle(capture) != source or _record(capture / "capture_receipt.json") != _record(out / "input-receipt.json"):
        raise ValueError("source effect capture changed during qualification")
    receipt = {"schema":QUALIFICATION,"source_directory":str(capture),"source_receipt":_record(out / "input-receipt.json"),
        "artifacts":{str(p.relative_to(out)):_record(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    _json(out / "qualification.json",receipt)
    try:
        verify_effect_qualification(out)
    except BaseException:
        (out / "qualification.json").unlink(missing_ok=True)
        raise
    return receipt


def verify_effect_qualification(out):
    out = Path(out)
    receipt = json.loads((out / "qualification.json").read_text())
    required = {"input-receipt.json","compiler/compile-inputs.json","compiler/program.mlir","compiler/signature.json",
        "host/model.so","host/model.ll","host/prepared.mlir","host/compilation.json","host/execution.json",
        "evaluator/inputs.npz","evaluator/reference.json","evaluation/outcome.json","evaluation/invocation-started.json",
        "evaluation/stdout.bin","evaluation/stderr.bin"}
    if receipt.get("schema") != QUALIFICATION or not required <= set(receipt.get("artifacts",{})):
        raise ValueError("incomplete Merlin effect qualification")
    for name,pin in receipt["artifacts"].items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name or name == "qualification.json"
                or any(out.joinpath(*relative.parts[:i]).is_symlink() for i in range(len(relative.parts)+1))
                or not (out / relative).is_file() or _record(out / relative) != pin):
            raise ValueError("effect qualification artifact integrity mismatch: "+name)
    source = json.loads((out / "input-receipt.json").read_text())
    stage = json.loads((out / "compiler/compile-inputs.json").read_text())
    if (source.get("schema") != SCHEMA or receipt.get("source_receipt") != _record(out / "input-receipt.json")
            or stage.get("source_profile") != "standalone-effect-kernel"
            or stage.get("source",{}).get("capture_receipt_sha256") != receipt["source_receipt"]["sha256"]
            or stage["source"].get("model_mlir_sha256") != source["artifacts"]["model.mlir"]["sha256"]):
        raise ValueError("effect compiler stage differs from its original source capture")
    for name,digest in stage["members"].items():
        if "compiler/"+name not in receipt["artifacts"] or _record(out / "compiler" / name)["sha256"] != digest:
            raise ValueError("effect compiler member differs from its byte binding")
    reference = json.loads((out / "evaluator/reference.json").read_text())
    validate_reference(reference)
    for name in ("inputs.npz",*reference_members(reference)):
        if "evaluator/"+name not in receipt["artifacts"] or _record(out / "evaluator" / name) != source["artifacts"].get(name):
            raise ValueError("initial effect qualification differs from its original reference case")
    compilation = json.loads((out / "host/compilation.json").read_text())
    if (compilation.get("schema") != "t2m.merlin-host-compilation.v1" or compilation.get("whole_body_lowering") != "passed"
            or compilation.get("compiler_input_staging") != "passed" or compilation.get("native_object_and_link") != "passed"
            or compilation.get("evaluator_data_supplied_to_compiler") is not False
            or compilation.get("source_program") != _record(out / "compiler/program.mlir")
            or compilation.get("prepared_program") != _record(out / "host/prepared.mlir")
            or compilation.get("llvm") != _record(out / "host/model.ll") or compilation.get("shared_library") != _record(out / "host/model.so")):
        raise ValueError("effect qualification lacks its actual whole-body compiler materialization")
    reference = json.loads((out / "evaluator/reference.json").read_text())
    trace = json.loads((out / "compiler/source/compiler-provenance.json").read_text())
    validate_reference(reference,trace)
    spec,signature = checked_signature(out / "compiler")
    if source.get("materialized_abi") != {"complete":True,"inputs":len(spec.inputs),"outputs":len(spec.outputs)}:
        raise ValueError("effect qualification changed the original public argument/result roster")
    comparison = _compare(out / "compiler",out / "host",out / "evaluation",out / "evaluator",reference,spec,signature)
    report = json.loads((out / "host/execution.json").read_text())
    if (report.get("schema") != "t2m.merlin-effect-host-qualification.v1" or any(report.get(k) != v for k,v in comparison.items())
            or report.get("host_execution") != "matched-expected-outcome" or report.get("reference") != _record(out / "evaluator/reference.json")
            or report.get("observation") != _record(out / "evaluation/outcome.json")):
        raise ValueError("Merlin effect comparison differs from its actual observed outcome")
    return receipt


def invoke_effect_kernel(qualification,arguments,reference,out,*,merlin_root,python=None):
    qualification,out = Path(qualification).resolve(),Path(out).absolute()
    original = verify_effect_qualification(qualification)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if qualification in out.parents:
        raise ValueError("effect invocation must be outside its immutable qualification")
    spec,signature = checked_signature(qualification / "compiler")
    expected = json.loads(Path(reference).read_text())
    trace = json.loads((qualification / "compiler/source/compiler-provenance.json").read_text())
    validate_reference(expected,trace)
    mixed = expected["schema"] == "t2m.effect-reference.v2"
    if expected["termination"] == "return" and spec.outputs and not mixed:
        raise ValueError("returned tensor values require numeric references")
    with np.load(arguments,allow_pickle=False) as arrays:
        if set(arrays.files) != {f"in{i}" for i in range(len(spec.inputs))}:
            raise ValueError("effect invocation requires the complete input roster")
        inputs = validate_inputs(spec,[arrays[f"in{i}"] for i in range(len(spec.inputs))])
    if mixed:
        validate_results(spec,inputs,load_archive(Path(reference).parent / ARCHIVE,len(spec.outputs)),
            expected["results"]["comparisons"])
    stage = json.loads((qualification / "compiler/compile-inputs.json").read_text())
    names = {"qualification.json","input-receipt.json","compiler/compile-inputs.json",
             "host/compilation.json","host/model.so","host/model.ll","host/prepared.mlir"} | {"compiler/"+name for name in stage["members"]}
    out.mkdir(parents=True)
    for name in names:
        target = out / "source" / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(qualification / name,target)
    evaluator = out / "evaluator"
    evaluator.mkdir()
    shutil.copyfile(arguments,evaluator / "inputs.npz")
    shutil.copyfile(reference,evaluator / "reference.json")
    if mixed:
        shutil.copyfile(Path(reference).parent / ARCHIVE,evaluator / ARCHIVE)
    observe_merlin_host(out / "source",evaluator / "inputs.npz",out / "evaluation",merlin_root=merlin_root,python=python)
    comparison = _compare(out / "source/compiler",out / "source/host",out / "evaluation",evaluator,expected,spec,signature)
    if verify_effect_qualification(qualification) != original:
        raise ValueError("effect qualification changed during invocation")
    report = {"schema":"t2m.merlin-effect-invocation.v1","native_compilation":"reused-qualified-library",**comparison,
        "source_qualification":_record(out / "source/qualification.json"),"reference":_record(evaluator / "reference.json"),
        "observation":_record(out / "evaluation/outcome.json"),
        "artifacts":{str(p.relative_to(out)):_record(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    _json(out / "invocation.json",report)
    try:
        verify_effect_invocation(out)
    except BaseException:
        (out / "invocation.json").unlink(missing_ok=True)
        raise
    return report


def verify_effect_invocation(out):
    out = Path(out)
    report = json.loads((out / "invocation.json").read_text())
    required = {"source/qualification.json","source/input-receipt.json","source/compiler/compile-inputs.json",
        "source/compiler/program.mlir","source/compiler/signature.json","source/host/model.so","source/host/model.ll",
        "source/host/prepared.mlir","source/host/compilation.json","evaluator/inputs.npz","evaluator/reference.json",
        "evaluation/outcome.json","evaluation/stdout.bin","evaluation/stderr.bin","evaluation/invocation-started.json"}
    if (report.get("schema") != "t2m.merlin-effect-invocation.v1" or not required <= set(report.get("artifacts",{}))
            or report.get("native_compilation") != "reused-qualified-library"):
        raise ValueError("incomplete reused effect invocation")
    for name,pin in report["artifacts"].items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name or name == "invocation.json"
                or any(out.joinpath(*relative.parts[:i]).is_symlink() for i in range(len(relative.parts)+1))
                or not (out / relative).is_file() or _record(out / relative) != pin):
            raise ValueError("effect invocation artifact integrity mismatch: "+name)
    source = out / "source"
    qualification = json.loads((source / "qualification.json").read_text())
    stage = json.loads((source / "compiler/compile-inputs.json").read_text())
    if qualification.get("schema") != QUALIFICATION or report.get("source_qualification") != _record(source / "qualification.json"):
        raise ValueError("effect invocation lacks its exact source qualification")
    names = {"input-receipt.json","compiler/compile-inputs.json","host/model.so","host/model.ll","host/prepared.mlir","host/compilation.json"}
    names.update("compiler/"+name for name in stage["members"])
    for name in names:
        if "source/"+name not in report["artifacts"] or qualification["artifacts"].get(name) != _record(source / name):
            raise ValueError("effect invocation snapshot differs from its original qualified artifacts")
    spec,signature = checked_signature(source / "compiler")
    expected = json.loads((out / "evaluator/reference.json").read_text())
    trace = json.loads((source / "compiler/source/compiler-provenance.json").read_text())
    validate_reference(expected,trace)
    if any("evaluator/"+name not in report["artifacts"] for name in reference_members(expected)):
        raise ValueError("effect invocation lacks its complete result reference artifacts")
    with np.load(out / "evaluator/inputs.npz",allow_pickle=False) as arrays:
        if set(arrays.files) != {f"in{i}" for i in range(len(spec.inputs))}:
            raise ValueError("effect invocation has an incomplete input roster")
        validate_inputs(spec,[arrays[f"in{i}"] for i in range(len(spec.inputs))])
    comparison = _compare(source / "compiler",source / "host",out / "evaluation",out / "evaluator",expected,spec,signature)
    if (any(report.get(k) != v for k,v in comparison.items()) or report.get("reference") != _record(out / "evaluator/reference.json")
            or report.get("observation") != _record(out / "evaluation/outcome.json")):
        raise ValueError("effect invocation does not bind its actual independent outcome comparison")
    return report
