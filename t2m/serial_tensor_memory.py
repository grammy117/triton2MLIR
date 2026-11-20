"""Scalar unstructured memory as ordered standard tensor values.

Reuse Meta 17f0c234 UnstructuredToMemref's scalar load/store event boundary
and the existing unit-tensor carrier. Unlike its physical reinterpret-cast,
this adapter uses the declared complete tensor root, active bounds and one
functional insertion. Parallel ownership and multi-lane scatters stay separate.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
# Licensed under the MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

from xdsl.dialects import arith, builtin, cf, scf, tensor
from xdsl.ir import Block, Region

from .diagnostics import ConversionError


def flat_storage_type(pointer, buffer, opcode):
    """Keep complete scalar-addressed roots linear across their serial versions.

    The public tensor shape remains unchanged. A root used by structured or
    multi-lane events retains its existing representation and ownership proof.
    """
    if len(buffer.shape) < 2 or all(size == 1 for size in buffer.shape) or not pointer.uses:
        return None
    for use in pointer.uses:
        op = use.operation
        if use.index != 0 or opcode(op) not in ("tts.gather", "tts.scatter"):
            return None
        if opcode(op) == "tts.scatter" and (not isinstance(op.operands[1].type, builtin.TensorType)
                or any(size != 1 for size in op.operands[1].type.get_shape())):
            return None
    count = builtin.DYNAMIC_INDEX if any(isinstance(size, str) for size in buffer.shape) else math.prod(buffer.shape)
    return builtin.TensorType(buffer.type.element_type, [count])


def collapse(value, target):
    group = builtin.ArrayAttr([builtin.ArrayAttr(builtin.IntegerAttr(axis, builtin.i64) for axis in range(len(value.type.get_shape())))])
    return tensor.CollapseShapeOp.build(operands=[value], result_types=[target], properties={"reassociation": group})


def restore(value, target, dimensions):
    shape = target.get_shape()
    group = builtin.ArrayAttr([builtin.ArrayAttr(builtin.IntegerAttr(axis, builtin.i64) for axis in range(len(shape)))])
    return tensor.ExpandShapeOp(value, dimensions, group, shape, target)


def unit_value(value, append):
    """Read one actual tensor element, retaining its original scalar SSA if available."""
    if not isinstance(value.type, builtin.TensorType) or any(size != 1 for size in value.type.get_shape()):
        raise ConversionError("serial scalar memory requires exactly one source lane")
    if isinstance(value.owner, tensor.FromElementsOp) and len(value.owner.operands) == 1:
        return value.owner.operands[0]
    zero = append(arith.ConstantOp.from_int_and_width(0, builtin.IndexType())).result
    return append(tensor.ExtractOp(value, [zero] * len(value.type.get_shape()), value.type.element_type)).result


def scatter(source, buffer, base, mapping, append, maximum_shape):
    """One predicated scalar write, preserving every other logical element."""
    def add(op):
        return append(op, source)

    if len(source.operands) not in (3, 4) or buffer.effect == "read":
        raise ConversionError("serial scatter needs writable storage and the exact index/value/mask schema", opcode="tts.scatter")
    index, value = (mapping[v] for v in source.operands[1:3])
    mask = mapping[source.operands[3]] if len(source.operands) == 4 else None
    if (not isinstance(index.type, builtin.TensorType) or index.type.element_type not in (builtin.i32, builtin.i64)
            or not isinstance(value.type, builtin.TensorType) or index.type.get_shape() != value.type.get_shape()
            or value.type.element_type != buffer.type.element_type
            or mask is not None and mask.type != builtin.TensorType(builtin.i1, index.type.get_shape())):
        raise ConversionError("serial scalar scatter has incompatible index/value/mask types", opcode="tts.scatter")
    raw = unit_value(index, add)
    scalar = unit_value(value, add)
    condition = unit_value(mask, add) if mask is not None else None
    maximum = math.prod(maximum_shape)
    if not 0 <= maximum < 2**31:
        raise ConversionError("serial scatter logical allocation needs a representable nonnegative 32-bit index domain", opcode="tts.scatter")
    shape = base.type.get_shape()
    group = builtin.ArrayAttr([builtin.ArrayAttr(builtin.IntegerAttr(axis, builtin.i64) for axis in range(len(shape)))])
    flat = base
    if len(shape) > 1:
        flat = add(tensor.CollapseShapeOp.build(operands=[base], result_types=[builtin.TensorType(base.type.element_type,
            [builtin.DYNAMIC_INDEX if any(size < 0 for size in shape) else maximum])], properties={"reassociation": group})).result
    if not shape:
        raise ConversionError("serial scalar scatter rank-zero storage needs a rank-aware root adapter", opcode="tts.scatter")
    zero_axis = add(arith.ConstantOp.from_int_and_width(0, builtin.IndexType())).result
    extent = add(tensor.DimOp(flat, zero_axis)).result

    def update(emit):
        offset = raw if raw.type == builtin.i64 else emit(arith.ExtSIOp(raw, builtin.i64)).result
        zero = emit(arith.ConstantOp.from_int_and_width(0, builtin.i64)).result
        upper = emit(arith.IndexCastOp(extent, builtin.i64)).result
        low = emit(arith.CmpiOp(offset, zero, "sge")).result
        high = emit(arith.CmpiOp(offset, upper, "slt")).result
        active = emit(arith.AndIOp(low, high)).result
        emit(cf.AssertOp(active, "Triton scalar scatter active index outside logical allocation"))
        coordinate = emit(arith.IndexCastOp(offset, builtin.IndexType())).result
        return emit(tensor.InsertOp(scalar, flat, [coordinate])).result

    if condition is None:
        updated = update(add)
    else:
        active, inactive = Block(), Block()

        def inside(op):
            op.location = source.location
            op.attributes.update({"prov.family": builtin.StringAttr("memory"), "prov.op": builtin.StringAttr("scalar_store")})
            active.add_op(op)
            return op

        active.add_op(scf.YieldOp(update(inside)))
        inactive.add_op(scf.YieldOp(flat))
        updated = add(scf.IfOp(condition, [flat.type], Region(active), Region(inactive))).results[0]
    if len(shape) > 1:
        sizes = [add(tensor.DimOp(base, add(arith.ConstantOp.from_int_and_width(axis, builtin.IndexType())).result)).result
                 for axis, size in enumerate(shape) if size < 0]
        updated = add(tensor.ExpandShapeOp(updated, sizes, group, shape, base.type)).result
    return updated, raw
