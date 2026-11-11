"""Portable source-byte custody for executable allocation-state bindings.

These checks bind retained compiler artifacts. They grant no source-equivalence,
numeric, alias, memory or target proof; the actual bound program still executes.
"""

import hashlib
import json

SCHEMA = "t2m.source-allocation-capsule.v1"
BASE = frozenset(
    {
        "allocation-original.ttir",
        "allocation-input.mlir",
        "allocation-arguments.json",
        "allocation-source-stage.json",
        "allocation-source-stderr.log",
    }
)
NORMALIZED = frozenset(
    {
        "allocation-normalized.ttir",
        "allocation-intake.json",
        "allocation-inventory.json",
    }
)


def record(raw):
    raw = raw.encode() if isinstance(raw, str) else raw
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def members(trace):
    binding = trace.get("source_argument_binding", {})
    capsule = binding.get("source_capsule")
    if capsule is None:
        if "descriptor_bindings" in binding:
            raise ValueError(
                "descriptor allocation bindings require portable source custody"
            )
        return frozenset()
    if (
        not isinstance(capsule, dict)
        or capsule.get("schema") != SCHEMA
        or type(capsule.get("normalized")) is not bool
        or capsule.get("grants_semantics") is not False
    ):
        raise ValueError("unknown or authority-granting source capsule profile")
    expected = BASE | (NORMALIZED if capsule["normalized"] else frozenset())
    if set(capsule.get("artifacts", {})) != expected:
        raise ValueError("source capsule lacks its complete compiler artifact roster")
    return expected


def validate(trace, artifacts, source):
    names = members(trace)
    if not names:
        return
    if not names <= set(artifacts):
        raise ValueError("source capsule is missing a declared compiler artifact")
    binding = trace["source_argument_binding"]
    capsule = binding["source_capsule"]
    if (
        binding.get("schema") != "t2m.source-allocation-bindings.v1"
        or binding.get("grants_semantics") is not False
        or binding.get("allocation_entry") != trace.get("spec", {}).get("entry")
        or binding.get("bound_source_sha256") != record(source)["sha256"]
        or binding.get("original_source_sha256")
        != record(artifacts["allocation-original.ttir"])["sha256"]
    ):
        raise ValueError("source capsule differs from original/bound allocation source")
    for name in names:
        if name not in artifacts or capsule["artifacts"][name] != record(
            artifacts[name]
        ):
            raise ValueError("source capsule artifact mismatch: " + name)
    arguments = json.loads(artifacts["allocation-arguments.json"])
    if (
        arguments.get("schema")
        not in (
            "t2m.source-allocation-arguments.v1",
            "t2m.source-allocation-arguments.v2",
        )
        or arguments.get("source_entry") != binding.get("source_entry")
        or not isinstance(arguments.get("arguments"), list)
    ):
        raise ValueError(
            "source capsule argument roster differs from its declared entry"
        )
    descriptor = binding.get("descriptor_bindings")
    if descriptor is not None and (
        not capsule["normalized"]
        or arguments["schema"] != "t2m.source-allocation-arguments.v2"
        or descriptor.get("schema") != arguments["schema"]
        or descriptor.get("grants_semantics") is not False
        or descriptor.get("source_entry") != binding["source_entry"]
        or descriptor.get("allocation_entry") != binding["allocation_entry"]
    ):
        raise ValueError(
            "descriptor allocation source custody differs from its declared argument interface"
        )
    input_name = (
        "allocation-normalized.ttir"
        if capsule["normalized"]
        else "allocation-original.ttir"
    )
    stage = json.loads(artifacts["allocation-source-stage.json"])
    expected = {
        "source.ttir": record(artifacts[input_name]),
        "lowered.mlir": record(artifacts["allocation-input.mlir"]),
        "stderr.log": record(artifacts["allocation-source-stderr.log"]),
    }
    if (
        set(stage)
        != {
            "schema",
            "source_stage_schema",
            "stage",
            "stage_completed",
            "flags",
            "artifacts",
            "optimizer",
            "grants_semantics",
        }
        or stage.get("schema") != "t2m.source-stage-custody.v1"
        or stage.get("source_stage_schema") != "t2m.triton-shared-stage.v1"
        or stage.get("grants_semantics") is not False
        or stage.get("stage") != "source"
        or stage.get("stage_completed") is not True
        or stage.get("artifacts") != expected
        or set(stage.get("optimizer", {})) != {"sha256", "bytes"}
        or stage.get("flags")
        != [
            "--mlir-disable-threading",
            "--verify-each",
            "--mlir-print-op-generic",
            "--mlir-print-debuginfo",
            "--mlir-print-local-scope",
        ]
    ):
        raise ValueError("source capsule does not bind the native generic source stage")
    optimizer = stage["optimizer"]
    if (
        type(optimizer["bytes"]) is not int
        or optimizer["bytes"] <= 0
        or not isinstance(optimizer["sha256"], str)
        or len(optimizer["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in optimizer["sha256"])
    ):
        raise ValueError("source capsule stage lacks an actual optimizer fingerprint")
    if capsule["normalized"]:
        intake = json.loads(artifacts["allocation-intake.json"])
        inventory = json.loads(artifacts["allocation-inventory.json"])
        allowed = {
            "schema",
            "triton_version",
            "pass",
            "pipeline",
            "target_backend",
            "normalized_opcodes",
            "source_sha256",
            "output_sha256",
            "dot_precision_implementation",
            "descriptor_implementation",
            "descriptor_profile",
            "descriptor_argument_expansion",
            "tf32_numeric_profile",
        }
        if (
            not set(intake) <= allowed
            or intake.get("schema") != "t2m.intake-pass.v1"
            or intake.get("target_backend") != "not-invoked"
            or intake.get("source_sha256")
            != record(artifacts["allocation-original.ttir"])["sha256"]
            or intake.get("output_sha256")
            != record(artifacts["allocation-normalized.ttir"])["sha256"]
            or inventory.get("normalization_passes") != intake.get("pipeline")
            or inventory.get("normalized_opcodes") != intake.get("normalized_opcodes")
        ):
            raise ValueError(
                "source capsule does not bind the original/normalized intake"
            )
