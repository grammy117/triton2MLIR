"""Select upstream's pointer-state prepass only when its state can be needed.

Meta 17f0c234's TritonToStructuredPass.cpp boxes integer tensor loop state,
then relies on canonicalization to remove boxes unrelated to pointers. The
installed 0c27923 binary boxes i64 data although its structured-state verifier
accepts only i32. Reuse its supported skip-prepass option when the control
interfaces are numeric and their boxed tensor state has no pointer/mask dependencies.
Independent ordinary memory in those regions stays visible. No region is rewritten,
outlined or hidden from pointer conversion or later memory/ownership checks.
"""

from xdsl.dialects import builtin

NUMERIC_TT = {"tt.splat", "tt.broadcast", "tt.expand_dims", "tt.trans",
    "tt.reshape", "tt.cat", "tt.join", "tt.split", "tt.dot", "tt.reduce",
    "tt.reduce.return", "tt.make_range", "tt.get_program_id", "tt.get_num_programs"}
CONTROL = {"scf.for", "scf.if", "scf.while", "scf.yield", "scf.condition"}
DIAGNOSTICS = {"tt.print", "tt.assert", "cf.assert"}
POINTER_MEMORY = {"tt.load", "tt.store", "tt.addptr", "tt.make_tensor_ptr", "tt.advance",
                  "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.trans", "tt.bitcast"}


def _numeric(type_):
    if isinstance(type_, builtin.TensorType):
        type_ = type_.element_type
    return isinstance(type_, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat))


def data_control_report(module, source_ids):
    """Return a complete conservative proof, or retain the original pipeline.

    A numeric recurrence may be stored as data or observed by diagnostics.
    Its transitive results may not reach pointer arithmetic, load/store masks,
    calls or unknown operations. Original ordinary accesses may occur inside
    a region when independent of every boxed tensor state. Scalar integer
    state needs no tensor structured-state boxing; its source-width and
    address checks remain independent.
    """
    from .shared_tensor import opcode

    def numeric_op(op):
        name = opcode(op)
        return ((name in NUMERIC_TT | CONTROL | DIAGNOSTICS
                 or name.split(".", 1)[0] in {"arith", "math", "tensor", "linalg"})
            and all(_numeric(v.type) for v in (*op.operands, *op.results))
            and all(len(region.blocks) == 1 and all(_numeric(arg.type) for arg in region.block.args)
                    for region in op.regions))

    # A result-free conditional carries no boxed pointer/numeric state. Its
    # memory operations still pass through the original pointer conversion.
    # Inspect any stateful control nested inside it independently.
    controls = [op for op in module.walk() if opcode(op) in {"scf.for", "scf.if", "scf.while"}
                and (op.results or opcode(op) != "scf.if")]
    if not controls or not any(isinstance(v.type, builtin.TensorType)
            and isinstance(v.type.element_type, builtin.IntegerType)
            for op in controls for v in (*op.operands, *op.results)):
        return None
    def supported(op):
        return numeric_op(op) or opcode(op) in POINTER_MEMORY and not op.regions

    if any(not numeric_op(op) or any(not supported(nested) for nested in op.walk()) for op in controls):
        return None

    def boxed(value):
        return isinstance(value.type, builtin.TensorType) and isinstance(value.type.element_type, builtin.IntegerType)

    pending = [value for op in controls for value in op.results if boxed(value)]
    for op in controls:
        for region in op.regions:
            for block in region.blocks:
                # The for induction variable is not boxed carried state.
                pending.extend(value for value in (block.args[1:] if opcode(op) == "scf.for" else block.args) if boxed(value))
    visited = set()
    while pending:
        value = pending.pop()
        if value in visited:
            continue
        visited.add(value)
        for use in value.uses:
            op = use.operation
            name = opcode(op)
            if name in DIAGNOSTICS or name == "tt.store" and use.index == 1:
                continue
            if not numeric_op(op) or any(not supported(nested) for nested in op.walk()):
                return None
            pending.extend(op.results)
            if name == "scf.yield" and opcode(op.parent_op()) in {"scf.for", "scf.if"}:
                pending.append(op.parent_op().results[use.index])
            elif name in {"scf.yield", "scf.condition"}:
                pending.extend(op.parent_op().results)

    return {"kind": "numeric-control-without-pointer-state",
        "analysis": "numeric-control-interfaces; ordinary-memory-retained; transitive-boxed-tensor-state-uses-data-only",
        "source_ids": list(dict.fromkeys(sid for op in controls for sid in source_ids(op.location))),
        "upstream_option": "skip-prepass=true",
        "scope": "pointer conversion, source-width and memory/ownership checks remain active"}
