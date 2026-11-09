# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Unsigned high products, using Ascend's standard arithmetic conversion.

Adapted from TritonMulhiuiConverter in TritonOpConverter.cpp at Ascend
8f0ade66580245e30ad97093526a4018e4e18fc7. The original converter selects
arith.mului_extended's high result. The tensor form here makes the parallel
elementwise body explicit for the shared tensor pipeline; it selects no ISA.
"""

from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError


def expand_high_products(module, source_ids):
    from .shared_tensor import opcode
    reports = []
    for op in list(module.walk()):
        if opcode(op) != "tt.mulhiui":
            continue
        ids = source_ids(op.location)
        type_ = op.results[0].type
        element = type_.element_type if isinstance(type_, builtin.TensorType) else type_
        if (not isinstance(element, builtin.IntegerType) or element.width.data not in (32, 64)
                or len(op.operands) != 2 or any(v.type != type_ for v in op.operands)):
            raise ConversionError("unsigned high products require matching scalar/tensor i32 or i64 words",
                                  opcode="tt.mulhiui", source_id=next(iter(ids), None))
        if isinstance(type_, builtin.TensorType):
            shape = type_.get_shape()
            if any(n <= 0 for n in shape):
                raise ConversionError("unsigned high product source tiles must have static positive extents",
                                      opcode="tt.mulhiui", source_id=next(iter(ids), None))
            body = Block(arg_types=[element] * 3)
            product = arith.MulUIExtendedOp(body.args[0], body.args[1])
            terminal = YieldOp(product.high)
            body.add_ops([product, terminal])
            empty = tensor.EmptyOp([], type_)
            expanded = GenericOp(list(op.operands), [empty.tensor], Region(body),
                [builtin.AffineMapAttr(AffineMap.identity(len(shape)))] * 3,
                [IteratorTypeAttr.parallel()] * len(shape), [type_])
            emitted, values = [empty, expanded], [expanded.results[0]]
            for child in (product, terminal):
                child.location = op.location
        else:
            expanded = arith.MulUIExtendedOp(*op.operands)
            emitted, values = [expanded], [expanded.high]
        expanded.attributes.update({"prov.composite_family": builtin.StringAttr("unsigned_high_product"),
            "prov.composite_id": builtin.StringAttr("umulhi:" + (ids[0] if ids else str(op.location))),
            "prov.source_word_bits": builtin.IntegerAttr(element.width.data, builtin.i32)})
        for generated in emitted:
            generated.location = op.location
        Rewriter.replace_op(op, emitted, values)
        reports.append({"kind": "ttir-unsigned-high-product", "source_ids": ids, "word_bits": element.width.data,
            "interpretation": "unsigned-source-bit-patterns", "expansion": "standard-mului-extended-high-result",
            "physical_mapping": "undecided"})
    return reports
