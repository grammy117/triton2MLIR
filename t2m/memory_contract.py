"""Byte-bound memory continuation evidence, distinct from numeric admission."""

import hashlib
import json
import math

TENSOR_PROFILE = "shared-functional-tensor"
BUFFER_PROFILE = "shared-functional-buffer"
GUARDED_PROFILE = "shared-guarded-buffer"
WORD_PROFILE = "shared-source-word-buffer"
GUARDED_PROFILES = (GUARDED_PROFILE, WORD_PROFILE)
MEMORY_FILES = {"memory-witness.mlir", "memory-input.mlir", "memory-stage.mlir", "memory-stage.json"}
GUARDED_FILES = MEMORY_FILES - {"memory-witness.mlir"} | {"memory-admission.json"}
WORD_FILES = {"word-input.mlir", "word-tensor.mlir", "word-stage.json"}
WORD_PIPELINE = ["--triton-to-unstructured=offset-bit-width=64",
    "--triton-arith-to-linalg=pids-to-func-args=false tt-to-func-func=false", "--remove-dead-values", "--canonicalize"]
PROFILE_FILES = {TENSOR_PROFILE: set(), BUFFER_PROFILE: MEMORY_FILES, GUARDED_PROFILE: GUARDED_FILES,
                 WORD_PROFILE: GUARDED_FILES | WORD_FILES}


def guard_layout(spec, *, atomics=False):
    """Complemented fragments with room for the selected ownership flags."""
    from .guard_protocol import ATOMIC, ORDINARY
    bits = (ATOMIC if atomics else ORDINARY).fragment_bits
    bounds = {s["name"]: s.get("bounds") for s in spec.get("runtime_scalars", [])}
    counts = [n if type(n) is int else (bounds[n["dimension"]][1] + n["divisor"] - 1) // n["divisor"]
              for n in spec.get("grid", [1])]
    words = max(1, ((math.prod(counts) - 1).bit_length() + bits - 1) // bits)
    if not 1 <= words <= 3:
        raise ValueError("guard owner layout exceeds the declared i64 logical grid")
    return {"owner_words_per_element": words, "guard_storage_bytes_per_output_element": 8 * (words + 1),
            **({"owner_fragment_bits": bits} if atomics else {})}


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
    members.append(("memory-admission.json", "admission_sha256") if profile in GUARDED_PROFILES else ("memory-witness.mlir", "witness_sha256"))
    for name, key in members:
        if memory.get(key) != record(artifacts[name])["sha256"]:
            raise ValueError("memory evidence differs from source provenance: " + name)
    stage = json.loads(artifacts["memory-stage.json"])
    admission = json.loads(artifacts["memory-admission.json"]) if profile in GUARDED_PROFILES else None
    version3 = admission is not None and admission.get("schema") == "t2m.memory-admission.v3"
    version2 = admission is not None and admission.get("schema") in ("t2m.memory-admission.v2", "t2m.memory-admission.v3")
    if (stage.get("schema") != "t2m.triton-shared-stage.v1" or stage.get("stage") != ("memory-effects" if version2 else "memory-ir")
            or stage.get("stage_completed") is not True
            or stage.get("artifacts", {}).get("source.ttir") != record(artifacts["memory-input.mlir"])
            or stage.get("artifacts", {}).get("lowered.mlir") != record(artifacts["memory-stage.mlir"])):
        raise ValueError("memory continuation report does not bind its actual input/output")
    if profile in GUARDED_PROFILES:
        spec = json.loads(json.dumps(trace.get("spec")))
        if not isinstance(spec, dict) or not isinstance(spec.get("buffers"), list):
            raise ValueError("guarded memory admission lacks its logical buffer ABI")
        layout = guard_layout(spec, atomics=version3) if version2 else {"guard_storage_bytes_per_output_element": 16}
        from .guard_protocol import ATOMIC, ORDINARY
        if (admission.get("schema") not in ("t2m.memory-admission.v1", "t2m.memory-admission.v2", "t2m.memory-admission.v3")
                or admission.get("strategy") != "runtime-guarded-private-output-state"
                or admission.get("spec") != spec or admission.get("internal_arguments") != guarded_arguments(spec)
                or admission.get("guarded_input_sha256") != record(artifacts["memory-input.mlir"])["sha256"]
                or admission.get("continuation_output_sha256") != record(artifacts["memory-stage.mlir"])["sha256"]
                or any(admission.get(key) != value for key, value in layout.items())
                or version2 and admission.get("ownership_protocol") != (ATOMIC if version3 else ORDINARY).name
                or admission.get("writable_reads") != ("shared-readers-or-single-writer-program" if version2 else "refused")
                or admission.get("fresh_initialization") != "post-grid-complete-write-coverage"):
            raise ValueError("guarded memory admission does not bind its source/ABI/continuation")
        buffers = {b["name"]: b for b in spec["buffers"]}
        records = trace.get("normalized_operations", trace.get("source_operations", []))
        inventory = {r["id"] for r in records}
        if not isinstance(admission.get("accesses"), list):
            raise ValueError("guarded memory admission lacks its access inventory")
        for access in admission["accesses"]:
            buffer = buffers.get(access.get("buffer"))
            atomic = version3 and access.get("opcode") in ("tts.unstructured_atomic_rmw", "tt.atomic_cas")
            write = access.get("opcode") in ("tts.store", "tts.scatter") or atomic
            allowed = ("tts.load", "tts.store", "tts.gather", "tts.scatter") + (("tts.unstructured_atomic_rmw", "tt.atomic_cas") if version3 else ())
            if (buffer is None or access.get("opcode") not in allowed
                    or access.get("effect") != ("atomic-readwrite" if atomic else "write" if write else "read")
                    or write and buffer["effect"] == "read"
                    or not write and buffer["effect"] != "read" and not version2
                    or not access.get("source_ids") or any(s not in inventory for s in access["source_ids"])
                    or not access.get("tile") or any(type(n) is not int or n <= 0 for n in access["tile"])):
                raise ValueError("guarded memory access inventory differs from its source/effect contract")
        if version3:
            from .atomic_memory import validate_evidence
            if profile != WORD_PROFILE:
                raise ValueError("atomic memory evidence requires its source-word address profile")
            validate_evidence(admission, memory, records, buffers)
    if profile == WORD_PROFILE:
        from .word_evidence import check_word_stage, replay_events
        word_stage = json.loads(artifacts["word-stage.json"])
        if (not version2 or admission.get("address_policy") != "source-words-offset64"
                or memory.get("address_policy") != "source-words-offset64"
                or memory.get("word_stage_sha256") != record(artifacts["word-stage.json"])["sha256"]
                or admission.get("word_input_sha256") != record(artifacts["word-input.mlir"])["sha256"]
                or admission.get("word_tensor_sha256") != record(artifacts["word-tensor.mlir"])["sha256"]):
            raise ValueError("source-word address evidence does not bind its declared input/output/policy")
        check_word_stage(word_stage, artifacts["word-input.mlir"], artifacts["word-tensor.mlir"])
        if memory.get("word_conversion") != ("native-isolated-events-v2" if word_stage.get("schema") == "t2m.source-word-stages.v2" else None):
            raise ValueError("source-word event conversion lacks its explicit representation difference")
        replay_events(word_stage, artifacts["word-input.mlir"])
    return profile
