"""Qualify execution-sized bound models through source-only Merlin staging.

An unqualified candidate lives under work, never at the publication path.
Actual descriptors precede reference comparison. The explicit v2 receipt names
the Merlin host backend; it does not claim a second native compiler oracle.
"""

import hashlib
import os
import shutil
from copy import deepcopy
from pathlib import Path

from xdsl.dialects import func
from xdsl.dialects.builtin import StringAttr

from .bundle import _json, _record
from .consumer import check_merlin_host, check_merlin_interface
from .executed_model_binding import serialize
from .model_binding import bind_model
from .model_package_members import retain_kernels, retain_parent

SCHEMA = "t2m.bound-model-receipt.v2"
PROFILE = "merlin.execution-owned-results.v1"
QUALIFICATION = "t2m.execution-owned-model-qualification.v1"
RETARGET = "relative canonical filename; payload bytes unchanged"


def inventory(root):
    return {
        str(path.relative_to(root)): _record(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "capture_receipt.json"
    }


def write(
    capture,
    bindings,
    out,
    *,
    work,
    python,
    merlin_root,
    merlin_python,
    merlin_host,
    merlin_clang,
    rtol,
    atol,
    execution_allocation_policy=None,
):
    from .model_bundle import (
        CORE,
        OPTIONAL_DATA,
        _arguments,
        _compare_results,
        _parent,
        _references,
    )

    if merlin_root is None or not merlin_host:
        raise ValueError(
            "execution-owned model qualification requires merlin_root and merlin_host=True"
        )
    from .execution_resources import checked_policy

    resource_policy = checked_policy(execution_allocation_policy)
    parent = _parent(capture)
    parent_receipt = _record(capture / "capture_receipt.json")
    bindings = list(bindings)
    bound = bind_model(
        (capture / "model.mlir").read_text(), bindings, result_mode="execution-owned"
    )
    trace = deepcopy(bound.trace)
    trace["before_weights_retarget_sha256"] = trace["mlir_sha256"]
    bound.module.attributes["prov.weights_file"] = StringAttr("weights.safetensors")
    source = serialize(bound.module)
    trace.update(
        mlir_sha256=hashlib.sha256(source.encode()).hexdigest(),
        weights_retarget=RETARGET,
    )
    work.mkdir(parents=True, exist_ok=False)
    candidate = work / "candidate"
    candidate.mkdir()
    parent_bindings = retain_parent(capture, parent, candidate, CORE | OPTIONAL_DATA)
    (candidate / "model.mlir").write_text(source)
    _json(candidate / "kernel-bindings.json", trace)
    kernels, profiles = retain_kernels(bindings, trace, candidate, located=True)
    compatibility = {
        "schema": "t2m.bound-model-compatibility.v2",
        "model_abi_and_core_data_files": "unchanged-from-parent-model2MLIR-capture",
        "kernel_profiles": profiles,
        "target_execution": "not-performed",
        "differences": [
            "t2m.bound-model-receipt.v2 explicitly selects execution-owned qualification; original m2m receipt and all parent archives remain byte-identical",
            "t2m.model-kernel-binding.v4 carries executable scalar/dimension/cast contracts and complete source/callsite provenance",
            "kernels/<ordinal>/located.mlir additionally retains source locations; its byte binding must reproduce the original unlocated implementation digest",
            "Merlin independently reconstructs the entire original model/kernel/adapter SSA before compile-signature.v6 staging; evaluator arrays and references stay excluded",
            "native qualification uses the actual Merlin host compiler and owned descriptor runtime with an independent original PyTorch reference; no separate torch-mlir execution oracle is claimed",
            "result dimensions come from executed descriptors; all source-domain, ownership and allocation-budget checks remain active",
            "the provisional source-only input receipt remains under consumer/input-receipt.json; only a complete qualified final receipt is published",
            "generated C, target harnesses, physical views and resizing ownership need separate qualification",
        ],
    }
    if "execution_view_binding_profile" in trace:
        compatibility["differences"].append(
            trace["execution_view_binding_profile"] + " extends v4 with explicit dense/source view records; "
            "Merlin independently reconstructs their source coordinates and standard adapters; "
            "older v4 readers refuse these fields and no physical alias or mutation permission is added"
        )
    if "source_layout_binding_profile" in trace:
        compatibility["differences"].append(
            trace["source_layout_binding_profile"] + " extends v4 with exact source-layout scalar bindings; "
            "Merlin independently checks the original typed call and full executable expansion; "
            "source packing geometry supplies no consumer hardware placement or physical alias permission"
        )
    if resource_policy is not None:
        compatibility["differences"].append(
            resource_policy["schema"] + " explicitly selects bounded allocation work and byte limits; "
            "the legacy default is unchanged and all source, allocation and lifetime checks remain active"
        )
    _json(candidate / "compatibility.json", compatibility)
    receipt = {
        "schema": SCHEMA,
        "result_profile": PROFILE,
        "qualification": "source-only-unqualified",
        "artifacts": inventory(candidate),
        "parent_artifact_bindings": parent_bindings,
        "parent_receipt": parent_receipt,
        "kernel_artifacts": kernels,
        "materialized_abi": "unchanged-from-parent",
        **({"execution_allocation_policy": resource_policy} if resource_policy is not None else {}),
    }
    _json(candidate / "capture_receipt.json", receipt)
    # Input materialization is a separate evaluator operation. No argument path
    # or array enters the source-only staging and compilation subprocesses.
    _inputs, sources = _arguments(
        capture,
        bound.module,
        work / "arguments",
        python,
        compiler_support_root=merlin_root,
        dynamic_inputs=True,
    )
    check_merlin_interface(
        candidate,
        candidate / "consumer",
        merlin_root=merlin_root,
        python=merlin_python or python,
        result_mode="execution-owned",
    )
    actual, host = check_merlin_host(
        candidate / "consumer",
        work / "arguments/arguments.npz",
        merlin_root=merlin_root,
        python=merlin_python or python,
        compiler_python=python,
        clang=merlin_clang or os.environ.get("T2M_MERLIN_CLANG"),
    )
    entry = next(
        op
        for op in bound.module.ops
        if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"
    )
    shapes = [list(value.shape) for value in actual]
    expected, reference_members = _references(
        capture, parent, entry.function_type.outputs, result_shapes=shapes
    )
    _compare_results(actual, expected, rtol, atol, entry.function_type.outputs)
    if (
        _parent(capture) != parent
        or _record(capture / "capture_receipt.json") != parent_receipt
    ):
        raise ValueError("parent capture changed during execution-owned qualification")
    host.update(
        schema="t2m.merlin-host-qualification.v2",
        host_execution="passed",
        reference_comparison="passed",
        reference="unchanged-parent-PyTorch-golden",
        reference_sha256=_record(candidate / "golden.npy")["sha256"],
        result_count=len(expected),
        result_reference_members=reference_members,
        rtol=rtol,
        atol=atol,
    )
    _json(candidate / "consumer/host/execution.json", host)
    execution = {
        "schema": QUALIFICATION,
        "result_profile": PROFILE,
        "execution": "Merlin-host-execution-owned",
        "input_sha256": trace["mlir_sha256"],
        "reference_comparison": "passed",
        "reference": "unchanged-parent-PyTorch-golden",
        "source_triton_executed": False,
        "independent_native_compiler_oracle": False,
        "rtol": rtol,
        "atol": atol,
        "materialized_arguments": sources,
        "result_count": len(expected),
        "result_shapes": shapes,
        "result_reference_members": reference_members,
        "result_observations": host["evaluation"]["result_observations"],
        "outputs": _record(candidate / "consumer/host/outputs.npz"),
        "compiled_library": _record(candidate / "consumer/host/model.so"),
    }
    _json(candidate / "execution.json", execution)
    _json(
        candidate / "meta.json",
        {
            "frontend": "model2MLIR+triton2MLIR",
            "entry": "forward",
            "qualification_backend": "Merlin-host-execution-owned",
            "native_model_reference_comparison": "passed",
            "merlin_host_execution": "passed",
            "merlin_target_execution": "not-performed",
        },
    )
    receipt.update(qualification="qualified", artifacts=inventory(candidate))
    _json(candidate / "capture_receipt.json", receipt)
    verify(candidate, receipt)
    shutil.copytree(candidate, out)
    verify(out, receipt)
    return out


def verify(out, receipt):
    from .executed_model_package_verify import verify as verify_package

    return verify_package(Path(out), receipt)
