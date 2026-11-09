"""Byte-bound memory continuation evidence, distinct from numeric admission."""

import hashlib
import json

TENSOR_PROFILE = "shared-functional-tensor"
BUFFER_PROFILE = "shared-functional-buffer"
GUARDED_PROFILE = "shared-guarded-buffer"
MEMORY_FILES = {"memory-witness.mlir", "memory-input.mlir", "memory-stage.mlir", "memory-stage.json"}
GUARDED_FILES = MEMORY_FILES - {"memory-witness.mlir"} | {"memory-admission.json"}
PROFILE_FILES = {TENSOR_PROFILE: set(), BUFFER_PROFILE: MEMORY_FILES, GUARDED_PROFILE: GUARDED_FILES}


def guarded_arguments(spec):
    return ([{"kind": "extent", "buffer": b["name"]} for b in spec["buffers"]]
            + [{"kind": kind, "buffer": b["name"]} for b in spec["buffers"] if b["effect"] != "read" for kind in ("owners", "visited")]
            + [{"kind": "program"}])


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
    members = [("memory-input.mlir", "continuation_input_sha256"), ("memory-stage.mlir", "continuation_output_sha256")]
    members.append(("memory-admission.json", "admission_sha256") if profile == GUARDED_PROFILE else ("memory-witness.mlir", "witness_sha256"))
    for name, key in members:
        if memory.get(key) != record(artifacts[name])["sha256"]:
            raise ValueError("memory evidence differs from source provenance: " + name)
    stage = json.loads(artifacts["memory-stage.json"])
    if (stage.get("schema") != "t2m.triton-shared-stage.v1" or stage.get("stage") != "memory-ir"
            or stage.get("stage_completed") is not True
            or stage.get("artifacts", {}).get("source.ttir") != record(artifacts["memory-input.mlir"])
            or stage.get("artifacts", {}).get("lowered.mlir") != record(artifacts["memory-stage.mlir"])):
        raise ValueError("memory continuation report does not bind its actual input/output")
    if profile == GUARDED_PROFILE:
        admission = json.loads(artifacts["memory-admission.json"])
        spec = json.loads(json.dumps(trace.get("spec")))
        if not isinstance(spec, dict) or not isinstance(spec.get("buffers"), list):
            raise ValueError("guarded memory admission lacks its logical buffer ABI")
        if (admission.get("schema") != "t2m.memory-admission.v1"
                or admission.get("strategy") != "runtime-guarded-private-output-state"
                or admission.get("spec") != spec or admission.get("internal_arguments") != guarded_arguments(spec)
                or admission.get("guarded_input_sha256") != record(artifacts["memory-input.mlir"])["sha256"]
                or admission.get("continuation_output_sha256") != record(artifacts["memory-stage.mlir"])["sha256"]
                or admission.get("guard_storage_bytes_per_output_element") != 16
                or admission.get("writable_reads") != "refused"
                or admission.get("fresh_initialization") != "post-grid-complete-write-coverage"):
            raise ValueError("guarded memory admission does not bind its source/ABI/continuation")
        buffers = {b["name"]: b for b in spec["buffers"]}
        inventory = {r["id"] for r in trace.get("normalized_operations", trace.get("source_operations", []))}
        if not isinstance(admission.get("accesses"), list):
            raise ValueError("guarded memory admission lacks its access inventory")
        for access in admission["accesses"]:
            buffer = buffers.get(access.get("buffer"))
            write = access.get("opcode") in ("tts.store", "tts.scatter")
            if (buffer is None or access.get("opcode") not in ("tts.load", "tts.store", "tts.gather", "tts.scatter")
                    or access.get("effect") != ("write" if write else "read")
                    or (buffer["effect"] == "read") == write
                    or not access.get("source_ids") or any(s not in inventory for s in access["source_ids"])
                    or not access.get("tile") or any(type(n) is not int or n <= 0 for n in access["tile"])):
                raise ValueError("guarded memory access inventory differs from its source/effect contract")
    return profile
