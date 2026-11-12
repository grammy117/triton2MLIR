"""Single-axis TTIR tile gather, adapted from Meta triton-shared 17f0c234.

Source: lib/Conversion/TensorGatherToLinalg/TensorGatherToLinalgPass.cpp,
TtGatherDerivedTensorGatherConverter. Apply its direct linalg.generic/extract
form to the actual source operation, without trusting a caller marker attribute.
Additional executable index guards precede every extraction.

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from xdsl.dialects import arith, builtin, cf, tensor
from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerAttr, TensorType
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, IndexOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError


def guarded_axis_index(word, extent, add):
    """Retain the same signed source-word guard for tensor and scalar gathers."""
    if word.type != builtin.i64:
        word = add(arith.ExtSIOp(word, builtin.i64))
    zero = add(arith.ConstantOp.from_int_and_width(0, builtin.i64))
    limit = add(arith.ConstantOp.from_int_and_width(extent, builtin.i64))
    low, high = (
        add(arith.CmpiOp(word, zero, "sge")),
        add(arith.CmpiOp(word, limit, "slt")),
    )
    add(
        cf.AssertOp(
            add(arith.AndIOp(low, high)), "TTIR tile gather index outside source axis"
        )
    )
    return add(arith.IndexCastOp(word, IndexType()))


def expand_tile_gathers(module, source_ids):
    from .shared_tensor import opcode

    reports = []
    for op in list(module.walk()):
        if opcode(op) != "tt.gather":
            continue
        ids = source_ids(op.location)

        def refuse(message):
            raise ConversionError(
                message, opcode="tt.gather", source_id=next(iter(ids), None)
            )

        if len(op.operands) != 2 or len(op.results) != 1:
            refuse("tile gather requires one source, indices and result")
        source, indices = op.operands
        types = [source.type, indices.type, op.results[0].type]
        if not all(isinstance(t, TensorType) for t in types):
            refuse("tile gather requires ranked tensor types")
        src_type, index_type, result_type = types
        shape = result_type.get_shape()
        axis = op.properties.get("axis")
        if (
            not isinstance(axis, IntegerAttr)
            or not 0 <= axis.value.data < len(shape)
            or index_type.get_shape() != shape
            or len(src_type.get_shape()) != len(shape)
            or any(n <= 0 for t in types for n in t.get_shape())
            or index_type.element_type not in (builtin.i32, builtin.i64)
            or result_type.element_type != src_type.element_type
            or not isinstance(
                src_type.element_type, (builtin.IntegerType, builtin.AnyFloat)
            )
        ):
            refuse(
                "tile gather requires static numeric tensors, matching result/indices and i32/i64 indices; pointer tensors need source-word normalization"
            )
        axis = axis.value.data
        if any(
            a != b
            for i, (a, b) in enumerate(zip(src_type.get_shape(), shape, strict=True))
            if i != axis
        ):
            refuse("tile gather non-axis dimensions must match")
        body = Block(arg_types=[index_type.element_type, result_type.element_type])

        def add(new):
            new.location = op.location
            body.add_op(new)
            return new.results[0] if new.results else None

        coordinate = guarded_axis_index(body.args[0], src_type.get_shape()[axis], add)
        coordinates = [
            coordinate if dim == axis else add(IndexOp(dim))
            for dim in range(len(shape))
        ]
        value = add(tensor.ExtractOp(source, coordinates, result_type.element_type))
        add(YieldOp(value))
        empty = tensor.EmptyOp([], result_type)
        empty.location = op.location
        generic = GenericOp(
            [indices],
            [empty.tensor],
            Region(body),
            [AffineMapAttr(AffineMap.identity(len(shape)))] * 2,
            [IteratorTypeAttr.parallel()] * len(shape),
            [result_type],
        )
        generic.location = op.location
        Rewriter.replace_op(op, [empty, generic])
        reports.append(
            {
                "kind": "ttir-tile-gather",
                "source_ids": ids,
                "axis": axis,
                "index_bits": index_type.element_type.width.data,
                "source_shape": list(src_type.get_shape()),
                "result_shape": list(shape),
                "expansion": "parallel-linalg-generic-with-tensor-extract",
                "bounds": "executable-axis-index-check",
                "physical_mapping": "undecided",
            }
        )
    return reports
