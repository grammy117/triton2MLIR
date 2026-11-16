"""Apply shared's standard elementwise cast pattern before region checking.

Adapted from LLVM ElementwiseToLinalg.cpp, which Meta/shared 17f0c234 and
archived Microsoft 050b93a register in their arithmetic converters. xDSL 0.68
binds these three casts to scalar integers only. Preserve the original scalar
operation, attributes and location inside an identity-mapped parallel generic;
do not weaken recursive SCF verification or change cast/poison semantics.
"""

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .scalar_tile_reads import numeric_tile


def expand_region_integer_casts(module, source_ids):
    from .shared_tensor import opcode

    plans = []
    for operation in module.walk():
        if not isinstance(operation, (arith.ExtSIOp, arith.ExtUIOp, arith.TruncIOp)):
            continue
        if not any(isinstance(value.type, builtin.TensorType)
                   for value in (*operation.operands, *operation.results)):
            continue
        parent = operation.parent_op()
        while parent is not None and opcode(parent) not in ("tt.reduce", "tt.scan", "tt.map_elementwise"):
            parent = parent.parent_op()
        if parent is None:
            continue
        source, target = operation.input.type, operation.result.type
        if (not numeric_tile(source) or not numeric_tile(target)
                or source.get_shape() != target.get_shape()
                or not isinstance(source.element_type, builtin.IntegerType)
                or not isinstance(target.element_type, builtin.IntegerType)
                or source.element_type.signedness.data != builtin.Signedness.SIGNLESS
                or target.element_type.signedness.data != builtin.Signedness.SIGNLESS
                or source.element_type.width.data <= 0 or target.element_type.width.data <= 0
                or ((source.element_type.width.data > target.element_type.width.data)
                    if isinstance(operation, arith.TruncIOp)
                    else (source.element_type.width.data < target.element_type.width.data)) is False):
            raise ConversionError("region integer cast must retain a positive static shape and its standard width direction",
                                  opcode=operation.name, source_id=next(iter(source_ids(operation.location)), None))
        plans.append((operation, source, target))
    for operation, source, target in plans:
        body = Block(arg_types=[source.element_type, target.element_type])
        scalar = type(operation).create(operands=[body.args[0]], result_types=[target.element_type],
                                        attributes=operation.attributes.copy(), properties=operation.properties.copy())
        scalar.verify()
        terminal = YieldOp(scalar.results[0])
        body.add_ops([scalar, terminal])
        empty = tensor.EmptyOp([], target)
        rank = len(source.get_shape())
        generic = GenericOp([operation.input], [empty.tensor], Region(body),
                            [builtin.AffineMapAttr(AffineMap.identity(rank))] * 2,
                            [IteratorTypeAttr.parallel()] * rank, [target])
        for child in (scalar, terminal, empty, generic):
            child.location = operation.location
        Rewriter.replace_op(operation, [empty, generic], list(generic.results))
