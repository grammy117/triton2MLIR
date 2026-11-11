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


def complete_parallel_rows(operation):
    """Prove pure tensor row updates from standard SSA, never annotations.

    Every parallel IV owns one entire output row, with no shared reads/writes
    through pointers. This admits ordinary value computation before guarded
    source-memory instrumentation; physical memory remains separately checked.
    """
    from xdsl.dialects import arith, scf, tensor
    from xdsl.traits import is_side_effect_free
    from .standard_ops import ForallOp, InParallelOp, ParallelInsertSliceOp
    if not isinstance(operation, ForallOp) or not value_only_region(operation):
        return False
    lower, upper, step = (tuple(getattr(operation, name).iter_values())
                          for name in ("staticLowerBound", "staticUpperBound", "staticStep"))
    if lower != (0,) or step != (1,) or len(upper) != 1 or upper[0] < 1 or any(
            (operation.dynamicLowerBound, operation.dynamicUpperBound, operation.dynamicStep)):
        return False
    block = operation.body.block
    end = block.last_op
    outputs = tuple(block.args[1:])
    if not outputs or not isinstance(end, InParallelOp):
        return False
    updates = tuple(end.body.block.ops)
    if len(updates) != len(outputs):
        return False
    seen = set()
    for update in updates:
        if (not isinstance(update, ParallelInsertSliceOp) or update.dest not in outputs or update.dest in seen
                or update.offsets != (block.args[0],) or update.sizes or update.strides):
            return False
        shape = update.dest.type.get_shape()
        if (len(shape) != 2 or shape[0] != upper[0] or shape[1] < 1
                or update.source.type.get_shape() != (shape[1],)
                or tuple(update.static_offsets.iter_values()) != (builtin.DYNAMIC_INDEX, 0)
                or tuple(update.static_sizes.iter_values()) != (1, shape[1])
                or tuple(update.static_strides.iter_values()) != (1, 1)):
            return False
        seen.add(update.dest)
    for child in operation.walk():
        if child is operation or child is end or child in updates or isinstance(child, (scf.IfOp, scf.ForOp, scf.WhileOp)):
            continue
        if not (is_side_effect_free(child) or isinstance(child, arith.ExtSIOp)):
            return False
    return True
