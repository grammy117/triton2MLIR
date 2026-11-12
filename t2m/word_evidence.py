"""Versioned source-word stage custody; no numerical-equivalence claim."""

import hashlib

POINTER_FLAGS = ["--triton-to-unstructured=offset-bit-width=64"]
COMPUTE_FLAGS = ["--triton-arith-to-linalg=pids-to-func-args=false tt-to-func-func=false",
                 "--remove-dead-values", "--canonicalize"]
RSQRT_COMPUTE_FLAGS = [COMPUTE_FLAGS[0], "--math-expand-ops=ops=rsqrt", *COMPUTE_FLAGS[1:]]
TRANSPORT = {"--mlir-disable-threading", "--verify-each", "--mlir-print-op-generic",
             "--mlir-print-debuginfo", "--mlir-print-local-scope"}


def record(raw):
    raw = raw.encode() if isinstance(raw, str) else raw
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def check_native(stage, name, flags, source, output, *, alternative_flags=()):
    if not isinstance(stage, dict):
        raise ValueError("source-word native stage must be an object")
    command = stage.get("command", [])
    pipeline = [flag for flag in (command if isinstance(command, list) else [])
                if isinstance(flag, str) and flag.startswith("--") and flag not in TRANSPORT]
    if (stage.get("schema") != "t2m.triton-shared-stage.v1" or stage.get("stage") != name
            or stage.get("stage_completed") is not True
            or not isinstance(command, list) or any(type(flag) is not str for flag in command)
            or pipeline != flags and pipeline not in alternative_flags
            or stage.get("artifacts", {}).get("source.ttir") != record(source)
            or stage.get("artifacts", {}).get("lowered.mlir") != record(output)):
        raise ValueError("source-word native stage does not bind its declared pipeline/input/output")


def check_word_stage(stage, source, output):
    if not isinstance(stage, dict):
        raise ValueError("source-word evidence must be an object")
    if stage.get("schema") == "t2m.triton-shared-stage.v1":
        check_native(stage, "source-word-tensor", POINTER_FLAGS + COMPUTE_FLAGS, source, output,
                     alternative_flags=(POINTER_FLAGS + RSQRT_COMPUTE_FLAGS,))
        return
    if (stage.get("schema") != "t2m.source-word-stages.v2" or stage.get("stage_completed") is not True
            or stage.get("input") != record(source) or stage.get("output") != record(output)
            or stage.get("reconstruction") != "same-region-same-event-order"
            or not isinstance(stage.get("events"), dict) or not isinstance(stage.get("compute"), dict)):
        raise ValueError("source-word event conversion lacks its versioned input/output chain")
    events, compute = stage["events"], stage["compute"]
    if (any(type(events.get(key)) is not str for key in ("input", "output"))
            or type(compute.get("input")) is not str or not isinstance(events.get("native_stage"), dict)
            or not isinstance(compute.get("native_stage"), dict) or not isinstance(events.get("bindings"), list)
            or not events["bindings"]):
        raise ValueError("source-word event conversion lacks its actual intermediate bytes")
    check_native(events["native_stage"], "source-word-events", POINTER_FLAGS, events["input"], events["output"])
    byte_views = [binding["byte_view"] for binding in events["bindings"]
                  if isinstance(binding, dict) and "byte_view" in binding]
    if (("byte_order" in events) != bool(byte_views)
            or byte_views and (events["byte_order"] not in ("little", "big")
                or any(not isinstance(view, dict) or view.get("byte_order") != events["byte_order"] for view in byte_views))):
        raise ValueError("source-word byte templates lack one explicit consistent byte order")
    check_native(compute["native_stage"], "source-word-compute", COMPUTE_FLAGS, compute["input"], output,
                 alternative_flags=(RSQRT_COMPUTE_FLAGS,))
    event_tool = events["native_stage"].get("optimizer", {}).get("sha256")
    if not isinstance(event_tool, str) or len(event_tool) != 64 or event_tool != compute["native_stage"].get("optimizer", {}).get("sha256"):
        raise ValueError("source-word event and computation stages used different optimizers")


def replay_events(stage, source):
    """Reconstruct every typed event disposition from the retained parent IR."""
    if stage.get("schema") != "t2m.source-word-stages.v2":
        return
    from .shared_tensor import _source_ids, parse_shared
    from .word_events import print_ir, substitute, templates
    module = parse_shared(source.decode() if isinstance(source, bytes) else source)
    events = stage["events"]
    isolated, bindings, calls = templates(module, _source_ids, byte_order=events.get("byte_order"))
    if print_ir(isolated) != events["input"] or bindings != events["bindings"]:
        raise ValueError("source-word event templates differ from the complete typed source event inventory")
    substitute(module, parse_shared(events["output"]), bindings, calls)
    if not same_computation(module, stage["compute"]["input"]):
        raise ValueError("source-word event reconstruction differs from the actual computation input")


def same_computation(module, expected):
    """Compare complete replay bodies across exact literal reification only.

    Older byte-bound stages may contain canonical dense NaNs. The executable
    parser now restores those literals as integer constants and bitcasts, so
    replay has that same representation. Reify only exact captured words in
    the retained expected body; no folding, dead-code removal, or other edit
    is permitted. All properties, attributes, locations and effects remain.
    Native stage input/output custody is checked separately before replay.
    """
    from .float_literals import materialize
    from .shared_tensor import parse_shared_source
    from .word_events import print_ir

    actual = print_ir(module)
    if actual == expected:
        return True
    original, literals = parse_shared_source(expected)
    if not literals:
        return False
    materialize(original, literals)
    return actual == print_ir(original)
