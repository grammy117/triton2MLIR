"""Executable shape bindings and unmapped logical launch sizes."""

from xdsl.dialects import arith, builtin, cf, tensor
from xdsl.dialects.builtin import IndexType, StringAttr

from .dynamic_abi import GridExtent


def emit(block, op, location, role):
    op.location = location
    op.attributes.update({"prov.family": StringAttr("abi"), "prov.op": StringAttr(role)})
    block.add_op(op)
    return op


def bind_dimensions(abi, values, block, location):
    dimensions = {}
    names = {n for b in abi.buffers for n in b.shape if isinstance(n, str)}
    names.update(g.dimension for g in abi.grid if isinstance(g, GridExtent))
    for name in sorted(names):
        value = values[name]
        if value.type != IndexType():
            value = emit(block, arith.IndexCastOp(value, IndexType()), location, "runtime-dimension").result
        dimensions[name] = value
    for buffer in abi.inputs:
        for axis, size in enumerate(buffer.shape):
            if not isinstance(size, str):
                continue
            index = emit(block, arith.ConstantOp.from_int_and_width(axis, IndexType()), location, "dimension-axis").result
            actual = emit(block, tensor.DimOp(values[buffer.name], index), location, "actual-tensor-dimension").result
            equal = emit(block, arith.CmpiOp(actual, dimensions[size], "eq"), location, "dimension-binding").result
            emit(block, cf.AssertOp(equal, "Triton runtime dimension differs from input shape: " + buffer.name), location, "dimension-binding")
    return dimensions


def launch_sizes(abi, dimensions, block, location):
    counts = []
    for extent in abi.grid:
        n = extent.divisor if isinstance(extent, GridExtent) else extent
        value = emit(block, arith.ConstantOp.from_int_and_width(n, IndexType()), location, "logical-launch-size").result
        if isinstance(extent, GridExtent):
            value = emit(block, arith.CeilDivUIOp(dimensions[extent.dimension], value), location, "runtime-logical-launch-size").result
        counts.append(value)
    return counts


def product(values, append):
    value = values[0]
    for other in values[1:]:
        value = append(arith.MuliOp(value, other)).result
    return value


def program_id(axis, iv, counts, append):
    if axis >= len(counts):
        return append(arith.ConstantOp.from_int_and_width(0, IndexType())).result
    value = iv
    if axis + 1 < len(counts):
        value = append(arith.DivUIOp(value, product(counts[axis + 1:], append))).result
    if len(counts) != 1:
        value = append(arith.RemUIOp(value, counts[axis])).result
    return value


def affine_index(form, iv, grid, counts, states, loops, append):
    """Materialize checked logical coordinates without dividing by shape sizes."""
    import math
    def constant(n):
        return append(arith.ConstantOp.from_int_and_width(n, IndexType())).result
    def index(value):
        return value if value.type == IndexType() else append(arith.IndexCastOp(value, IndexType())).result
    value = constant(form.constant)
    for axis, coefficient in enumerate(form.programs[:len(grid)]):
        if not coefficient or grid[axis] == 1:
            continue
        if counts is not None:
            pid = program_id(axis, iv, counts, append)
        else:
            pid = iv
            divisor = math.prod(grid[axis + 1:])
            if divisor != 1:
                pid = append(arith.DivUIOp(pid, constant(divisor))).result
            pid = append(arith.RemUIOp(pid, constant(grid[axis]))).result
        term = append(arith.MuliOp(pid, constant(coefficient))).result
        value = append(arith.AddiOp(value, term)).result
    for name, _, _, coefficient in form.parameters:
        term = append(arith.MuliOp(index(states[name]), constant(coefficient))).result
        value = append(arith.AddiOp(value, term)).result
    for identity, _, _, coefficient in form.loops:
        if identity not in loops:
            from .diagnostics import ConversionError
            raise ConversionError("logical coordinate escapes its proved loop scope")
        current, lower, step = loops[identity]
        ordinal = index(current)
        if lower:
            ordinal = append(arith.SubiOp(ordinal, constant(lower))).result
        if step != 1:
            ordinal = append(arith.DivSIOp(ordinal, constant(step))).result
        term = append(arith.MuliOp(ordinal, constant(coefficient))).result
        value = append(arith.AddiOp(value, term)).result
    return value
