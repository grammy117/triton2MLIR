"""Observable-outcome packages for void and initially failing tensor kernels."""

import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np

from .bundle import _json, _record
from .effect_reference import compare_outcome, validate_inputs, validate_reference
from .memory_contract import PROFILE_FILES, validate_memory_evidence
from .native import observe_native
from .package_compatibility import kernel_representation_differences
from .shared_tensor import TensorABI

SCHEMA = "t2m.effect-kernel-receipt.v1"
CORE = {"model.mlir","source.ttir","spec.json","provenance.json","inputs.npz","input_order.json",
        "weights.safetensors","weights.safetensors.manifest.json","extra.npz","reference.json",
        "compatibility.json","meta.json","execution.json"}
NATIVE = {"model.mlir","inputs.npz","lowered-llvm.mlir","invocation-started.json","observe-config.json",
          "stdout.bin","stderr.bin","outcome.json"}


def _native_files(spec, reference):
    return NATIVE | ({"shape-abi.json"} if spec.dynamic else set()) | (
        {"outputs.npz","execution.json"} if reference["termination"] == "return" else set())


def _comparison(root, spec, reference, trace):
    validate_reference(reference,trace)
    return compare_outcome(root / "native",reference,backend="native-jit",program=_record(root / "model.mlir"),
        arguments=_record(root / "inputs.npz"),implementation={"kind":"llvm-jit","artifact":_record(root / "native/lowered-llvm.mlir")},
        result_count=len(spec.outputs))


def write_effect_bundle(result, out, inputs, reference, *, work, python=None, compiler_support_root=None):
    out,work = Path(out),Path(work)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    reference = validate_reference(reference,result.trace)
    if reference["termination"] == "return" and result.spec.outputs:
        raise ValueError("successful tensor-returning kernels require numeric result references")
    arrays = validate_inputs(result.spec,inputs)
    if (hashlib.sha256(result.mlir_text.encode()).hexdigest() != result.trace["mlir_sha256"]
            or hashlib.sha256(result.source_ttir.encode()).hexdigest() != result.trace["source_sha256"]
            or json.loads(json.dumps(result.spec.as_dict())) != json.loads(json.dumps(result.trace["spec"]))):
        raise ValueError("effect source/payload/ABI changed after conversion")
    profile = validate_memory_evidence(result.trace,result.auxiliary_artifacts)
    observe_native(result.mlir_text,arrays,work,python=python,shape_abi=result.spec.runtime_shapes(),compiler_support_root=compiler_support_root)
    comparison = compare_outcome(work,reference,backend="native-jit",program=_record(work / "model.mlir"),arguments=_record(work / "inputs.npz"),
        implementation={"kind":"llvm-jit","artifact":_record(work / "lowered-llvm.mlir")},result_count=len(result.spec.outputs))
    out.mkdir(parents=True)
    (out / "model.mlir").write_text(result.mlir_text)
    (out / "source.ttir").write_text(result.source_ttir)
    for name in _native_files(result.spec,reference):
        target = out / "native" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(work / name,target)
    shutil.copyfile(work / "inputs.npz",out / "inputs.npz")
    _json(out / "spec.json",result.spec.as_dict())
    _json(out / "provenance.json",result.trace)
    _json(out / "reference.json",reference)
    (out / "weights.safetensors").write_bytes(struct.pack("<Q",8)+b"{}      ")
    _json(out / "weights.safetensors.manifest.json",{str(i):{"kind":"input","name":value.name} for i,value in enumerate(result.spec.inputs)})
    _json(out / "input_order.json",{value.name:i for i,value in enumerate(result.spec.inputs)})
    np.savez(out / "extra.npz")
    for name,data in result.auxiliary_artifacts.items():
        (out / name).write_text(data)
    _json(out / "compatibility.json",{"schema":"t2m.effect-kernel-compatibility.v1",
        "differences":["explicit effect-kernel receipt; reference.json contains an independent termination/stream outcome instead of golden.npy",
            "void entries have no tensor results or artificial output storage; failed tensor entries retain their declared result ABI and publish no values",
            "native observation retains exact byte streams, termination and invocation-phase bindings; this is producer observation, not source-equivalence attestation",
            "reference and observation artifacts are evaluator-only; compiler-only staging must exclude their values and digests",
            *kernel_representation_differences(result.trace,dynamic=result.spec.dynamic)]})
    _json(out / "meta.json",{"frontend":"triton2MLIR","entry":result.spec.entry,"outcome_reference":"observable-effects",
        "source_closure_verified":False,"target_execution":"not-performed"})
    _json(out / "execution.json",{"schema":"t2m.effect-execution.v1","input_sha256":result.trace["mlir_sha256"],
        "reference":_record(out / "reference.json"),"observation":_record(out / "native/outcome.json"),**comparison,
        "source_kernel_execution":"not-performed","consumer_execution":"not-performed"})
    required = CORE | set(PROFILE_FILES[profile]) | {"native/"+name for name in _native_files(result.spec,reference)}
    _json(out / "capture_receipt.json",{"schema":SCHEMA,"profile":profile,
        "artifacts":{name:_record(out / name) for name in sorted(required)},
        "materialized_abi":{"complete":True,"inputs":len(result.spec.inputs),"outputs":len(result.spec.outputs)},
        "outcome_reference":"reference.json","source_closure_verified":False})
    try:
        verify_effect_bundle(out)
    except BaseException:
        (out / "capture_receipt.json").unlink(missing_ok=True)
        raise
    return out


def verify_effect_bundle(out):
    out = Path(out)
    receipt = json.loads((out / "capture_receipt.json").read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("profile") not in PROFILE_FILES:
        raise ValueError("not an effect-kernel capture receipt")
    for name,pin in receipt.get("artifacts",{}).items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name
                or any(out.joinpath(*relative.parts[:i]).is_symlink() for i in range(len(relative.parts)+1))
                or not (out / relative).is_file() or _record(out / relative) != pin):
            raise ValueError("effect artifact integrity mismatch: "+name)
    spec = TensorABI.from_dict(json.loads((out / "spec.json").read_text()))
    reference,trace = (json.loads((out / name).read_text()) for name in ("reference.json","provenance.json"))
    required = CORE | set(PROFILE_FILES[receipt["profile"]]) | {"native/"+name for name in _native_files(spec,reference)}
    if set(receipt.get("artifacts",{})) != required or receipt.get("outcome_reference") != "reference.json":
        raise ValueError("effect receipt lacks its complete explicit outcome profile")
    if (trace.get("spec") != json.loads(json.dumps(spec.as_dict())) or trace.get("mlir_sha256") != _record(out / "model.mlir")["sha256"]
            or trace.get("source_sha256") != _record(out / "source.ttir")["sha256"]):
        raise ValueError("effect provenance differs from its source/payload/ABI")
    if validate_memory_evidence(trace,{name:(out / name).read_text() for name in PROFILE_FILES[receipt["profile"]]}) != receipt["profile"]:
        raise ValueError("effect receipt differs from its actual memory profile")
    if receipt.get("materialized_abi") != {"complete":True,"inputs":len(spec.inputs),"outputs":len(spec.outputs)}:
        raise ValueError("effect receipt differs from its actual declared ABI")
    manifest = {str(i):{"kind":"input","name":value.name} for i,value in enumerate(spec.inputs)}
    if ((out / "weights.safetensors").read_bytes() != struct.pack("<Q",8)+b"{}      "
            or json.loads((out / "weights.safetensors.manifest.json").read_text()) != manifest
            or json.loads((out / "input_order.json").read_text()) != {value.name:i for i,value in enumerate(spec.inputs)}):
        raise ValueError("effect input/weight bindings differ from the standalone ABI")
    with np.load(out / "inputs.npz",allow_pickle=False) as data:
        if set(data.files) != {f"in{i}" for i in range(len(spec.inputs))}:
            raise ValueError("effect input archive differs from the complete argument roster")
        validate_inputs(spec,[data[f"in{i}"] for i in range(len(spec.inputs))])
    with np.load(out / "extra.npz",allow_pickle=False) as data:
        if data.files:
            raise ValueError("effect standalone extra archive must be empty")
    if (_record(out / "native/model.mlir") != _record(out / "model.mlir")
            or _record(out / "native/inputs.npz") != _record(out / "inputs.npz")):
        raise ValueError("native effect observation differs from its actual program/arguments")
    comparison = _comparison(out,spec,reference,trace)
    execution = json.loads((out / "execution.json").read_text())
    if (execution.get("schema") != "t2m.effect-execution.v1" or any(execution.get(k) != v for k,v in comparison.items())
            or execution.get("input_sha256") != trace["mlir_sha256"] or execution.get("reference") != _record(out / "reference.json")
            or execution.get("observation") != _record(out / "native/outcome.json")):
        raise ValueError("effect reference comparison does not bind the actual observed outcome")
    return receipt
