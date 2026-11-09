"""Byte-bound memory continuation evidence, distinct from numeric admission."""

import hashlib
import json

TENSOR_PROFILE = "shared-functional-tensor"
BUFFER_PROFILE = "shared-functional-buffer"
MEMORY_FILES = {"memory-witness.mlir", "memory-input.mlir", "memory-stage.mlir", "memory-stage.json"}
PROFILE_FILES = {TENSOR_PROFILE: set(), BUFFER_PROFILE: MEMORY_FILES}


def validate_memory_evidence(trace, artifacts):
    profile = trace.get("profile", TENSOR_PROFILE)
    if profile not in PROFILE_FILES or set(artifacts) != PROFILE_FILES[profile]:
        raise ValueError("memory profile has an incomplete or unknown evidence inventory")
    if profile == TENSOR_PROFILE:
        return profile
    def record(raw):
        if isinstance(raw, str):
            raw = raw.encode()
        return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    memory = trace.get("memory", {})
    for name, key in (("memory-witness.mlir", "witness_sha256"), ("memory-input.mlir", "continuation_input_sha256"),
                      ("memory-stage.mlir", "continuation_output_sha256")):
        if memory.get(key) != record(artifacts[name])["sha256"]:
            raise ValueError("memory evidence differs from source provenance: " + name)
    stage = json.loads(artifacts["memory-stage.json"])
    if (stage.get("schema") != "t2m.triton-shared-stage.v1" or stage.get("stage") != "memory-ir"
            or stage.get("stage_completed") is not True
            or stage.get("artifacts", {}).get("source.ttir") != record(artifacts["memory-input.mlir"])
            or stage.get("artifacts", {}).get("lowered.mlir") != record(artifacts["memory-stage.mlir"])):
        raise ValueError("memory continuation report does not bind its actual input/output")
    return profile
