"""Derive a checked model bundle without forging a model2MLIR receipt."""

from copy import deepcopy
import hashlib
import json
from math import prod
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from xdsl.dialects import func
from xdsl.dialects.builtin import StringAttr, TensorType

from .bundle import _json, _record
from .consumer import check_merlin_interface
from .ir import text
from .model_binding import bind_model
from .native import execute_native

CORE = {"model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
        "input_order.json", "inputs.npz", "extra.npz", "golden.npy"}
ITEM_BYTES = {"f32": 4, "f64": 8, "i64": 8, "i32": 4, "i16": 2, "i8": 1, "i1": 1}


def _parent(capture):
    receipt_path = capture / "capture_receipt.json"
    if receipt_path.is_symlink():
        raise ValueError("parent capture receipt must be a direct regular file")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "m2m.capture-receipt.v1" or not CORE <= set(receipt.get("artifacts", {})):
        raise ValueError("expected a complete model2MLIR capture with its original receipt")
    for name, recorded in receipt["artifacts"].items():
        if Path(name).name != name or name == "capture_receipt.json":
            raise ValueError("parent artifact inventory must contain direct filenames")
        path = capture / name
        if path.is_symlink() or not path.is_file() or _record(path) != recorded:
            raise ValueError("parent capture integrity mismatch: " + name)
    return receipt


def _arguments(capture, module, work, python):
    entry = [op for op in module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"]
    if len(entry) != 1:
        raise ValueError("model bundle requires exactly one @forward entry")
    abi = []
    for type_ in entry[0].function_type.inputs:
        if not isinstance(type_, TensorType) or any(n < 0 for n in type_.get_shape()):
            raise ValueError("native model qualification requires static ranked tensor arguments")
        dtype = str(type_.get_element_type())
        if dtype not in ITEM_BYTES:
            raise ValueError(f"native model qualification does not yet support {dtype}")
        abi.append({"shape": list(type_.get_shape()), "dtype": dtype})
    if sum(prod(a["shape"]) * ITEM_BYTES[a["dtype"]] for a in abi) > 64 * 1024 * 1024:
        raise ValueError("materialized model arguments exceed the 64MiB qualification budget")
    if len(entry[0].function_type.outputs) != 1:
        raise ValueError("this model package profile requires one independently captured model golden")
    work.mkdir(parents=True, exist_ok=False)
    _json(work / "abi.json", abi)
    try:
        process = subprocess.run([str(python or sys.executable), str(Path(__file__).with_name("model_inputs.py")),
                                  str(capture.resolve()), str(work.resolve())], capture_output=True,
                                 text=True, timeout=30,
                                 env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                          TMPDIR=str(work.resolve())))
    except subprocess.TimeoutExpired as exc:
        raise ValueError("model argument materialization exceeded 30 seconds") from exc
    if process.returncode:
        raise ValueError("model argument materialization failed: " + process.stderr[-4000:])
    with np.load(work / "arguments.npz", allow_pickle=False) as data:
        values = [data[f"in{i}"].copy() for i in range(len(abi))]
    return values, json.loads((work / "bindings.json").read_text())


def write_model_bundle(capture, bindings, out, *, work, python=None, merlin_root=None,
                       merlin_python=None, rtol=0, atol=0):
    capture, out, work = Path(capture), Path(out), Path(work)
    if out.exists():
        raise FileExistsError(out)
    if not np.isfinite(rtol) or not np.isfinite(atol) or min(rtol, atol) < 0:
        raise ValueError("model comparison requires finite nonnegative tolerances")
    parent = _parent(capture)
    parent_receipt_record = _record(capture / "capture_receipt.json")
    bindings = list(bindings)
    bound = bind_model((capture / "model.mlir").read_bytes().decode("utf-8"), bindings)
    trace = deepcopy(bound.trace)
    trace["before_weights_retarget_sha256"] = trace["mlir_sha256"]
    bound.module.attributes["prov.weights_file"] = StringAttr("weights.safetensors")
    mlir = text(bound.module)
    trace["mlir_sha256"] = hashlib.sha256(mlir.encode()).hexdigest()
    trace["weights_retarget"] = "relative canonical filename; payload bytes unchanged"
    work.mkdir(parents=True, exist_ok=False)
    inputs, sources = _arguments(capture, bound.module, work / "arguments", python)
    (actual,), execution = execute_native(mlir, inputs, work / "native", python=python)
    golden = np.load(capture / "golden.npy", allow_pickle=False)
    if actual.shape != golden.shape or actual.dtype != golden.dtype:
        raise ValueError("captured golden differs from the complete model result ABI")
    np.testing.assert_allclose(actual, golden, rtol=rtol, atol=atol, equal_nan=True)
    if _parent(capture) != parent or _record(capture / "capture_receipt.json") != parent_receipt_record:
        raise ValueError("parent capture changed during qualification")
    execution.update({"reference_comparison": "passed", "reference": "unchanged-parent-PyTorch-golden",
                      "rtol": rtol, "atol": atol, "materialized_arguments": sources})
    out.mkdir(parents=True, exist_ok=False)
    (out / "parent").mkdir()
    shutil.copyfile(capture / "capture_receipt.json", out / "parent/capture_receipt.json")
    parent_bindings = {}
    for name in parent["artifacts"]:
        destination = name if name in CORE and name != "model.mlir" else "parent/" + name
        shutil.copyfile(capture / name, out / destination)
        parent_bindings[name] = destination
    (out / "model.mlir").write_text(mlir)
    _json(out / "kernel-bindings.json", trace)
    _json(out / "execution.json", execution)
    selected = {r["implementation_sha256"] for r in trace["calls"]}
    implementations = {b.implementation.trace["mlir_sha256"]: b.implementation for b in bindings
                       if b.implementation.trace["mlir_sha256"] in selected}
    kernel_artifacts = {}
    for i, (digest, implementation) in enumerate(implementations.items()):
        directory = out / "kernels" / str(i)
        directory.mkdir(parents=True)
        (directory / "source.ttir").write_text(implementation.source_ttir)
        (directory / "model.mlir").write_text(implementation.mlir_text)
        _json(directory / "spec.json", implementation.spec.as_dict())
        _json(directory / "provenance.json", implementation.trace)
        kernel_artifacts[digest] = str(directory.relative_to(out))
    if merlin_root is not None:
        check_merlin_interface(out, out / "consumer", merlin_root=merlin_root, python=merlin_python or python)
    _json(out / "compatibility.json", {"schema": "t2m.bound-model-compatibility.v1",
          "model_abi_and_core_data_files": "unchanged-from-parent-model2MLIR-capture",
          "differences": ["t2m.bound-model-receipt.v1; original m2m receipt retained under parent/",
                          "original FX trace remains under parent/ and binds original model bytes",
                          "kernel-bindings.json binds translated TTIR bodies and callsites",
                          "model.mlir has relative weights path and no unresolved registered kernel calls"],
          "target_execution": "not-performed"})
    _json(out / "meta.json", {"frontend": "model2MLIR+triton2MLIR", "entry": "forward",
          "native_model_reference_comparison": "passed", "merlin_target_execution": "not-performed"})
    artifacts = {str(p.relative_to(out)): _record(p) for p in sorted(out.rglob("*")) if p.is_file()}
    receipt = {"schema": "t2m.bound-model-receipt.v1", "artifacts": artifacts,
               "parent_artifact_bindings": parent_bindings, "parent_receipt": parent_receipt_record,
               "kernel_artifacts": kernel_artifacts, "materialized_abi": "unchanged-from-parent"}
    _json(out / "capture_receipt.json", receipt)
    verify_model_bundle(out)
    return out


def verify_model_bundle(out):
    out = Path(out)
    receipt = json.loads((out / "capture_receipt.json").read_text())
    required = CORE | {"parent/capture_receipt.json", "kernel-bindings.json", "execution.json", "compatibility.json", "meta.json"}
    if receipt.get("schema") != "t2m.bound-model-receipt.v1" or not required <= set(receipt.get("artifacts", {})):
        raise ValueError("not a complete bound model receipt")
    for name, expected in receipt["artifacts"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name == "capture_receipt.json":
            raise ValueError("invalid bound artifact path")
        path = out / relative
        parts = [out.joinpath(*relative.parts[:i]) for i in range(len(relative.parts) + 1)]
        if any(p.is_symlink() for p in parts) or not path.is_file() or _record(path) != expected:
            raise ValueError("bound bundle integrity mismatch: " + name)
    parent_path = out / "parent/capture_receipt.json"
    parent = json.loads(parent_path.read_text())
    if _record(parent_path) != receipt["parent_receipt"] or set(parent["artifacts"]) != set(receipt["parent_artifact_bindings"]):
        raise ValueError("incomplete parent receipt binding")
    for old_name, destination in receipt["parent_artifact_bindings"].items():
        if destination not in receipt["artifacts"] or _record(out / destination) != parent["artifacts"][old_name]:
            raise ValueError("parent artifact relation failed: " + old_name)
    trace = json.loads((out / "kernel-bindings.json").read_text())
    execution = json.loads((out / "execution.json").read_text())
    model_hash = _record(out / "model.mlir")["sha256"]
    if trace["mlir_sha256"] != model_hash or execution["input_sha256"] != model_hash or execution["reference_comparison"] != "passed":
        raise ValueError("binding/native validation does not bind the final model")
    for digest, directory in receipt["kernel_artifacts"].items():
        if directory + "/model.mlir" not in receipt["artifacts"]:
            raise ValueError("kernel implementation is outside the bound artifact inventory")
        if _record(out / directory / "model.mlir")["sha256"] != digest:
            raise ValueError("kernel implementation relation failed")
    return receipt
