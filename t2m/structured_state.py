"""Select upstream's pointer-state prepass only when its state can be needed.

Meta 17f0c234's TritonToStructuredPass.cpp boxes integer tensor loop state,
then relies on canonicalization to remove boxes unrelated to pointers. The
installed 0c27923 binary boxes i64 data although its structured-state verifier
accepts only i32. Reuse its supported skip-prepass option when the entire
control-flow state is demonstrably numeric. No source region is rewritten,
outlined or hidden from pointer conversion or later memory/ownership checks.
"""

from xdsl.dialects import builtin


NUMERIC_TT = {"tt.splat", "tt.broadcast", "tt.expand_dims", "tt.trans",
    "tt.reshape", "tt.cat", "tt.join", "tt.split", "tt.dot", "tt.reduce",
    "tt.reduce.return", "tt.make_range", "tt.get_program_id", "tt.get_num_programs"}
CONTROL = {"scf.for", "scf.if", "scf.while", "scf.yield", "scf.condition"}
DIAGNOSTICS = {"tt.print", "tt.assert", "cf.assert"}


def _numeric(type_):
    if isinstance(type_, builtin.TensorType):
        type_ = type_.element_type
    return isinstance(type_, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat))


def data_control_report(module, source_ids):
    """Return a complete conservative proof, or retain the original pipeline.

    A numeric recurrence may be stored as data or observed by diagnostics.
    Its transitive results may not reach pointer arithmetic, load/store masks,
    calls, unknown operations or another region with buffer access. In
    particular, a scalar integer address cast is not numeric data here.
    """
    from .shared_tensor import opcode

    def numeric_op(op):
        name = opcode(op)
        return ((name in NUMERIC_TT | CONTROL | DIAGNOSTICS
                 or name.split(".", 1)[0] in {"arith", "math", "tensor", "linalg"})
            and all(_numeric(v.type) for v in (*op.operands, *op.results))
            and all(len(region.blocks) == 1 and all(_numeric(arg.type) for arg in region.block.args)
                    for region in op.regions))

    controls = [op for op in module.walk() if opcode(op) in {"scf.for", "scf.if", "scf.while"}]
    if not controls or not any(isinstance(v.type, builtin.TensorType)
            and isinstance(v.type.element_type, builtin.IntegerType)
            for op in controls for v in (*op.operands, *op.results)):
        return None
    if any(not numeric_op(nested) for op in controls for nested in op.walk()):
        return None

    pending = [value for op in controls for value in op.results]
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
            if not numeric_op(op) or any(not numeric_op(nested) for nested in op.walk()):
                return None
            pending.extend(op.results)
            if name in {"scf.yield", "scf.condition"}:
                pending.extend(op.parent_op().results)

    return {"kind": "numeric-control-without-pointer-state",
        "analysis": "all-control-regions-numeric; transitive-state-uses-data-only",
        "source_ids": list(dict.fromkeys(sid for op in controls for sid in source_ids(op.location))),
        "upstream_option": "skip-prepass=true",
        "scope": "pointer conversion, source-width and memory/ownership checks remain active"}
