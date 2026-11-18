"""Reopen execution-owned model packages without a compiler or evaluator run.

Producer replay checks source expansion. Merlin owns its independent staging
reader. Product records attest byte custody and observed reference comparisons,
not arbitrary source equivalence or accelerator correctness.
"""

import hashlib
import json
import math
from copy import deepcopy

import numpy as np
from xdsl.dialects import func
from xdsl.dialects.builtin import StringAttr

from .bundle import _record
from .executed_model_binding import serialize
from .executed_model_package import PROFILE, QUALIFICATION, RETARGET, SCHEMA
from .memory_contract import PROFILE_FILES, TENSOR_PROFILE, validate_memory_evidence
from .model_binding import KernelBinding, bind_model
from .model_package_members import checked_members
from .shared_tensor import SharedTensorResult, TensorABI, parse_shared
from .source_capsule import members as source_members
from .source_capsule import validate as validate_capsule


def _reconstruct(out, receipt, trace):
    artifacts = receipt["artifacts"]
    if set(receipt["kernel_artifacts"]) != set(trace.get("kernels", {})) or set(
        receipt["kernel_artifacts"]
    ) != {record["implementation_sha256"] for record in trace["calls"]}:
        raise ValueError(
            "executed package implementation inventory differs from correspondence"
        )
    implementations = {}
    for digest, directory in receipt["kernel_artifacts"].items():
        if directory.split("/")[:1] != ["kernels"] or len(directory.split("/")) != 2:
            raise ValueError(
                "executed implementations require distinct kernel directories"
            )
        root = out / directory
        mandatory = {
            "model.mlir",
            "located.mlir",
            "source.ttir",
            "spec.json",
            "provenance.json",
        }
        if any(directory + "/" + name not in artifacts for name in mandatory):
            raise ValueError("executed package lacks a retained kernel compiler member")
        provenance = json.loads((root / "provenance.json").read_text())
        spec = json.loads((root / "spec.json").read_text())
        if (
            provenance != trace["kernels"][digest]
            or provenance.get("spec") != spec
            or _record(root / "model.mlir")["sha256"] != digest
            or provenance.get("mlir_sha256") != digest
            or provenance.get("source_sha256")
            != _record(root / "source.ttir")["sha256"]
            or trace.get("located_implementations", {}).get(digest)
            != _record(root / "located.mlir")
        ):
            raise ValueError("executed kernel source/ABI/located custody differs")
        names = PROFILE_FILES[
            provenance.get("profile", TENSOR_PROFILE)
        ] | source_members(provenance)
        if any(directory + "/" + name not in artifacts for name in names):
            raise ValueError(
                "executed kernel lacks its complete memory/source evidence"
            )
        auxiliary = {name: (root / name).read_text() for name in names}
        validate_memory_evidence(provenance, auxiliary)
        validate_capsule(provenance, auxiliary, (root / "source.ttir").read_text())
        module = parse_shared((root / "located.mlir").read_text())
        implementations[digest] = SharedTensorResult(
            module,
            TensorABI.from_dict(spec),
            provenance,
            (root / "source.ttir").read_text(),
            auxiliary,
        )
    bindings, selected = [], set()
    for record in trace["calls"]:
        fields = {
            "qualname": record["kernel"],
            "implementation": record["implementation_sha256"],
            "operands": record["source_operand_bindings"],
            "results": record["source_result_order"],
            "scalar_arguments": record["captured_scalar_arguments"],
            "runtime_arguments": record["runtime_scalar_arguments"],
            "runtime_operands": record["runtime_operand_arguments"],
            "source_layout_arguments": record["source_layout_arguments"],
        }
        if "tensor_views" in record:
            fields["tensor_views"] = record["tensor_views"]
        if "mutation_pointer_arguments" in record:
            fields["mutation_arguments"] = record["mutation_pointer_arguments"]
        if "source_tensor_views" in record:
            if "source_view_policy" in record:
                fields["source_view_policy"] = record["source_view_policy"]
            fields["source_views"] = {
                name: value["binding"]
                for name, value in record["source_tensor_views"].items()
            }
        key = json.dumps(fields, sort_keys=True)
        if key in selected:
            continue
        selected.add(key)
        fields["implementation"] = implementations[fields["implementation"]]
        fields["results"] = tuple(fields["results"])
        bindings.append(KernelBinding(**fields))
    rebound = bind_model(
        (out / "parent/model.mlir").read_text(), bindings, result_mode="execution-owned"
    )
    expected = deepcopy(rebound.trace)
    expected["before_weights_retarget_sha256"] = expected["mlir_sha256"]
    rebound.module.attributes["prov.weights_file"] = StringAttr("weights.safetensors")
    source = serialize(rebound.module)
    expected.update(
        mlir_sha256=hashlib.sha256(source.encode()).hexdigest(),
        weights_retarget=RETARGET,
    )
    if (
        json.loads(json.dumps(expected)) != trace
        or source != (out / "model.mlir").read_text()
    ):
        raise ValueError(
            "executed package differs from original source/body/guard reconstruction"
        )
    return rebound.module


def verify(out, receipt):
    try:
        return _verify(out, receipt)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("malformed execution-owned model package") from error


def _verify(out, receipt):
    from .model_bundle import CORE, _compare_results, _references

    required = CORE | {
        "parent/capture_receipt.json",
        "kernel-bindings.json",
        "execution.json",
        "compatibility.json",
        "meta.json",
        "consumer/input-receipt.json",
        "consumer/compiler/compile-inputs.json",
        "consumer/compiler/program.mlir",
        "consumer/compiler/signature.json",
        "consumer/host/execution.json",
        "consumer/host/compilation.json",
        "consumer/host/evaluation.json",
        "consumer/host/outputs.npz",
        "consumer/host/model.so",
        "consumer/host/owned-result-program.json",
    }
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("result_profile") != PROFILE
        or receipt.get("qualification") != "qualified"
        or receipt.get("materialized_abi") != "unchanged-from-parent"
    ):
        raise ValueError("not a qualified execution-owned bound model receipt")
    parent = checked_members(out, receipt, required)
    if (
        parent.get("schema") != "m2m.capture-receipt.v1"
        or parent.get("materialized_abi", {}).get("complete") is not True
    ):
        raise ValueError("executed model lacks its complete original parent ABI")
    trace = json.loads((out / "kernel-bindings.json").read_text())
    module = _reconstruct(out, receipt, trace)
    entry = next(
        op
        for op in module.ops
        if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"
    )
    execution = json.loads((out / "execution.json").read_text())
    tolerance = [execution.get(name) for name in ("rtol", "atol")]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in tolerance
    ):
        raise ValueError("executed model reference tolerance is invalid")
    if (
        execution.get("schema") != QUALIFICATION
        or execution.get("result_profile") != PROFILE
        or execution.get("execution") != "Merlin-host-execution-owned"
        or execution.get("independent_native_compiler_oracle") is not False
        or execution.get("input_sha256") != _record(out / "model.mlir")["sha256"]
        or execution.get("reference_comparison") != "passed"
        or execution.get("source_triton_executed") is not False
    ):
        raise ValueError(
            "executed model qualification does not bind its actual backend/source"
        )
    expected, members = _references(
        out,
        parent,
        entry.function_type.outputs,
        metadata=out / "parent/meta.json",
        result_shapes=execution.get("result_shapes"),
    )
    if (
        execution.get("result_count") != len(expected)
        or execution.get("result_reference_members") != members
    ):
        raise ValueError(
            "executed model qualification does not bind every original result"
        )
    with np.load(out / "consumer/host/outputs.npz", allow_pickle=False) as saved:
        keys = [f"out{i}" for i in range(len(expected))]
        if set(saved.files) != set(keys) or len(saved.files) != len(keys):
            raise ValueError("executed model observation omits a result")
        actual = [saved[key].copy() for key in keys]
    _compare_results(actual, expected, *tolerance, entry.function_type.outputs)
    if execution.get("result_shapes") != [list(value.shape) for value in actual]:
        raise ValueError(
            "executed model shape observations differ from its actual outputs"
        )
    staged = json.loads((out / "consumer/compiler/compile-inputs.json").read_text())
    for name, digest in staged["members"].items():
        if (
            "consumer/compiler/" + name not in receipt["artifacts"]
            or _record(out / "consumer/compiler" / name)["sha256"] != digest
        ):
            raise ValueError("executed model staged member differs from its manifest")
    if (
        staged.get("result_mode") != PROFILE
        or staged["source"]["model_mlir_sha256"] != trace["mlir_sha256"]
        or staged["source"]["capture_receipt_sha256"]
        != _record(out / "consumer/input-receipt.json")["sha256"]
    ):
        raise ValueError(
            "executed staging does not bind its original source-only input receipt"
        )
    signature = json.loads((out / "consumer/compiler/signature.json").read_text())
    if "execution_allocation_policy" in receipt:
        from .execution_resources import checked_policy

        selected_policy = checked_policy(receipt["execution_allocation_policy"])
        input_receipt = json.loads((out / "consumer/input-receipt.json").read_text())
        if (
            selected_policy is None
            or signature.get("allocation_policy") != selected_policy
            or input_receipt.get("execution_allocation_policy") != selected_policy
        ):
            raise ValueError("executed allocation policy differs from its original compiler selection")
    elif signature.get("allocation_policy", {}).get("schema") != "merlin.owned-result-limits.v1":
        raise ValueError("executed allocation policy lacks its explicit source-package selection")
    host = json.loads((out / "consumer/host/execution.json").read_text())
    compilation = json.loads((out / "consumer/host/compilation.json").read_text())
    evaluation = json.loads((out / "consumer/host/evaluation.json").read_text())
    saved_program = json.loads((out / "consumer/host/owned-result-program.json").read_text())
    if (
        saved_program.get("limits") != {
            key: signature["allocation_policy"][key]
            for key in ("max_bytes", "max_allocations", "max_output_bytes")
        }
        or compilation.get("owned_result_program") != _record(out / "consumer/host/owned-result-program.json")
    ):
        raise ValueError("executed saved program changes its selected allocation limits")
    if (
        signature.get("schema") != "merlin.compile-signature.v6"
        or signature.get("result_abi") != PROFILE
        or compilation.get("schema") != "t2m.merlin-host-compilation.v2"
        or compilation.get("result_abi") != PROFILE
        or compilation.get("whole_body_lowering") != "passed"
        or compilation.get("evaluator_data_supplied_to_compiler") is not False
        or compilation.get("source_program")
        != _record(out / "consumer/compiler/program.mlir")
        or compilation.get("source_signature")
        != _record(out / "consumer/compiler/signature.json")
        or compilation.get("shared_library") != _record(out / "consumer/host/model.so")
        or evaluation.get("schema") != "t2m.merlin-host-evaluation.v2"
        or evaluation.get("result_abi") != PROFILE
        or evaluation.get("functional_inputs_preserved") is not True
        or evaluation.get("invocations") != 2
        or host
        != {
            **compilation,
            "evaluation": evaluation,
            "schema": "t2m.merlin-host-qualification.v2",
            "host_execution": "passed",
            "reference_comparison": "passed",
            "reference": "unchanged-parent-PyTorch-golden",
            "reference_sha256": _record(out / "golden.npy")["sha256"],
            "result_count": len(expected),
            "result_reference_members": members,
            "rtol": tolerance[0],
            "atol": tolerance[1],
        }
        or execution.get("outputs") != _record(out / "consumer/host/outputs.npz")
        or evaluation.get("outputs") != execution["outputs"]
        or execution.get("compiled_library") != compilation["shared_library"]
        or execution.get("result_observations") != evaluation.get("result_observations")
        or len(execution["result_observations"]) != 2
        or any(
            row.get("source_calls") != 1
            or row.get("shapes") != execution["result_shapes"]
            for row in execution["result_observations"]
        )
    ):
        raise ValueError(
            "executed model host evidence differs from its source/library/result observations"
        )
    compatibility = json.loads((out / "compatibility.json").read_text())
    metadata = json.loads((out / "meta.json").read_text())
    if (
        compatibility.get("schema") != "t2m.bound-model-compatibility.v2"
        or metadata.get("qualification_backend") != "Merlin-host-execution-owned"
    ):
        raise ValueError(
            "executed model package omits its explicit compatibility/backend profile"
        )
    return receipt
