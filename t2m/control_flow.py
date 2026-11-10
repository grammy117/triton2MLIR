"""Recognize structured value computation that needs no buffer-state rewrite."""

from xdsl.dialects import builtin


def value_only_region(operation):
    """Exclude buffer accesses, pointers, opaque operations and CFG branches.

    Standard tensor operations compute immutable values. In particular, an
    extract from a previously loaded tensor is not another source buffer read.
    Assertions and scalar/string printing retain their executable semantics; this is not an effect-free
    classification. Source-memory regions still require the memory analysis.
    """
    def value_type(type_):
        if isinstance(type_, builtin.TensorType):
            type_ = type_.element_type
        return isinstance(type_, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat))

    for op in operation.walk():
        if op.name.split(".")[0] not in {"arith", "math", "tensor", "linalg", "scf", "cf"} and op.name != "vector.print":
            return False
        if op.name in {"cf.br", "cf.cond_br"}:
            return False
        values = [*op.operands, *op.results,
                  *(arg for region in op.regions for block in region.blocks for arg in block.args)]
        if not all(value_type(value.type) for value in values):
            return False
    return True
