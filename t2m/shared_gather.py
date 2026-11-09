"""Tensor-ABI adaptation of triton-shared's per-lane gather expansion.

Pattern: Meta triton-shared 17f0c234, UnstructuredToMemref/GatherConverter.
Retain its parallel generic and predicated extraction, replacing its memref
bridge with an immutable logical tensor and checking live offsets first.
Masked loads without `other` remain refused until definedness is propagated.
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

from xdsl.dialects import arith, builtin, cf, scf, tensor
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DenseArrayBase, DYNAMIC_INDEX, IndexType, IntegerAttr, IntegerType, StringAttr, TensorType
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, IndexOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from .diagnostics import ConversionError


def gather(source, buffer, base, mapping, append, *, maximum_shape=None):
    """Only read-only allocations: repeated/cross-program reads need no owner."""
    def refuse(message):
        raise ConversionError(message, opcode="tts.gather")

    if buffer.effect != "read":
        refuse("irregular reads from writable buffers need an initialization/ownership proof")
    segments = source.properties.get("operandSegmentSizes")
    if not isinstance(segments, DenseArrayBase):
        refuse("gather requires explicit operand segments")
    segments = tuple(segments.get_values())
    if len(segments) != 4 or segments[:2] != (1, 1) or any(n not in (0, 1) for n in segments[2:]) or sum(segments) != len(source.operands):
        refuse("gather has an unsupported operand schema")
    index = mapping[source.operands[1]]
    mask = mapping[source.operands[2]] if segments[2] else None
    other = mapping[source.operands[-1]] if segments[3] else None
    return _gather(source, buffer, base, index, mask, other, append,
                   maximum_shape=maximum_shape)


def structured_gather(source, pointer, buffer, base, mapping, append, *, maximum_shape=None):
    """Keep the upstream rank-one gather descriptor and exact prefix predicate."""
    def array(op, key):
        value = op.properties.get(key)
        if not isinstance(value, DenseArrayBase):
            raise ConversionError("structured gather needs typed descriptor arrays", opcode="tts.load")
        return tuple(value.get_values())

    dimension = pointer.properties.get("gather_scatter_dim")
    sizes = array(pointer, "sizes")
    if (len(sizes) != 1 or not isinstance(dimension, builtin.IntegerAttr) or dimension.value.data != 0
            or array(pointer, "operandSegmentSizes") != (1, 1, 0, 0, 0)
            or array(pointer, "static_strides") != (1,) or array(pointer, "static_offsets") != (0,)):
        raise ConversionError("structured gather needs its rank-one unit-stride zero-base descriptor", opcode="tts.load")
    segments, mask = array(source, "operandSegmentSizes"), array(source, "static_mask_dims")
    if (len(segments) != 3 or segments[0] != 1 or any(n not in (0, 1) for n in segments[1:])
            or sum(segments) != len(source.operands) or len(mask) > 1
            or segments[1] != int(bool(mask) and mask[0] == DYNAMIC_INDEX)):
        raise ConversionError("structured gather load has an unsupported prefix schema", opcode="tts.load")
    prefix = mapping[source.operands[1]] if segments[1] else mask[0] if mask else None
    other = mapping[source.operands[-1]] if segments[2] else None
    if prefix == sizes[0]:
        prefix = None
    return _gather(source, buffer, base, mapping[pointer.operands[1]], None, other, append, prefix=prefix,
                   maximum_shape=maximum_shape)


def _gather(source, buffer, base, index, mask, other, append, *, prefix=None, maximum_shape=None):
    def refuse(message):
        raise ConversionError(message, opcode="tts.gather")

    if buffer.effect != "read":
        refuse("irregular reads from writable buffers need an initialization/ownership proof")
    type_ = source.results[0].type
    if not isinstance(type_, TensorType) or not isinstance(index.type, TensorType):
        refuse("irregular gather needs ranked tensor indices and results")
    shape, scalar = type_.get_shape(), type_.get_element_type()
    element = index.type.get_element_type()
    if (any(n <= 0 for n in shape) or index.type.get_shape() != shape or not isinstance(element, IntegerType)
            or element.width.data not in (32, 64) or base.type.get_element_type() != scalar):
        refuse("gather requires matching static shapes and signed i32/i64 indices")
    dynamic = any(isinstance(n, str) for n in buffer.shape)
    if dynamic and maximum_shape is None:
        refuse("dynamic gather allocations require a bounded runtime allocation contract")
    maximum = 1
    for size in reversed(maximum_shape if dynamic else buffer.shape):
        maximum *= size
        if not 0 <= maximum < 2**31:
            refuse("gather logical allocation/strides need a representable nonnegative 32-bit index domain")
    if mask is not None and (mask.type != TensorType(builtin.i1, shape) or other is None):
        refuse("masked gather needs a matching boolean tensor and explicit defined fill")
    if other is not None and other.type != scalar:
        refuse("gather fill must be the declared scalar element type")
    if prefix is not None and (len(shape) != 1 or other is None):
        refuse("prefix-masked gather needs a rank-one tensor and explicit defined fill")
    # Match upstream's collapsed immutable base tensor. Dense ABI storage is a
    # view: do not decode every source flat offset into row/column coordinates.
    if len(buffer.shape) > 1:
        reassociation = ArrayAttr([ArrayAttr(IntegerAttr(i, builtin.i64) for i in range(len(buffer.shape)))])
        flat_type = TensorType(scalar, [DYNAMIC_INDEX if dynamic else maximum])
        base = append(tensor.CollapseShapeOp.build(operands=[base], result_types=[flat_type],
            properties={"reassociation": reassociation}), source).result
    extent = maximum
    if dynamic:
        zero = append(arith.ConstantOp.from_int_and_width(0, IndexType()), source).result
        extent = append(tensor.DimOp(base, zero), source).result
    inputs = [index, *([mask] if mask is not None else [])]
    block = Block(arg_types=[element, *([builtin.i1] if mask is not None else []), scalar])

    def load(body):
        def add(op):
            body.add_op(op)
            return op.results[0] if op.results else None
        offset = block.args[0]
        if offset.type != builtin.i64:
            offset = add(arith.ExtSIOp(offset, builtin.i64))
        zero = add(arith.ConstantOp.from_int_and_width(0, builtin.i64))
        upper = add(arith.ConstantOp.from_int_and_width(extent, builtin.i64)) if isinstance(extent, int) else add(arith.IndexCastOp(extent, builtin.i64))
        low_ok, high_ok = add(arith.CmpiOp(offset, zero, "sge")), add(arith.CmpiOp(offset, upper, "slt"))
        add(cf.AssertOp(add(arith.AndIOp(low_ok, high_ok)), "Triton gather active index outside logical allocation"))
        coordinate = add(arith.IndexCastOp(offset, IndexType()))
        return add(tensor.ExtractOp(base, [coordinate], scalar))

    condition = block.args[1] if mask is not None else None
    if prefix is not None:
        if isinstance(prefix, int):
            prefix = append(arith.ConstantOp.from_int_and_width(prefix, IndexType()), source).result
        lane = IndexOp(0)
        selected = arith.CmpiOp(lane.result, prefix, "slt")
        block.add_ops([lane, selected])
        condition = selected.result
    if condition is None:
        block.add_op(YieldOp(load(block)))
    else:
        active, inactive = Block(), Block()
        active.add_op(scf.YieldOp(load(active)))
        inactive.add_op(scf.YieldOp(other))
        choice = scf.IfOp(condition, [scalar], Region(active), Region(inactive))
        block.add_ops([choice, YieldOp(choice.results[0])])
    empty = append(tensor.EmptyOp([], type_), source).tensor
    generic = GenericOp(inputs, [empty], Region(block),
        [AffineMapAttr(AffineMap.identity(len(shape)))] * (len(inputs) + 1),
        [IteratorTypeAttr.parallel()] * len(shape), [type_])
    for op in generic.walk():
        op.location = source.location
        op.attributes.update({"prov.family": StringAttr("memory"), "prov.op": StringAttr("indexed_read")})
    return append(generic, source).results[0]
