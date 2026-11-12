# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Retain immutable tile lookups inside scalar computation.

Reuse Meta/shared 17f0c234's UnsplatConverter (tensor.extract at zero) and
the guarded direct single-axis gather in tile_gather. Fuse only the actual
one-element gather/unsplat value chain, without moving a load or inferring
source memory stability from annotations.
"""

from math import prod

from xdsl.dialects import arith, builtin, cf, tensor
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .tile_gather import guarded_axis_index


def numeric_tile(type_):
    return (
        isinstance(type_, builtin.TensorType)
        and isinstance(type_.element_type, (builtin.IntegerType, builtin.AnyFloat))
        and isinstance(type_.encoding, builtin.NoneAttr)
        and all(n > 0 for n in type_.get_shape())
    )


def scalar_tensor_read(operation):
    """An actual immutable tensor read, with no pointer or effect admission."""
    return (
        isinstance(operation, tensor.ExtractOp)
        and numeric_tile(operation.tensor.type)
        and len(operation.indices) == len(operation.tensor.type.get_shape())
        and operation.result.type == operation.tensor.type.element_type
        and all(
            isinstance(value.type, builtin.IndexType) for value in operation.indices
        )
    )


def expand_scalar_tile_reads(module, source_ids, *, generated_guards=None):
    from .shared_tensor import opcode

    plans = []
    for original in module.walk():
        if opcode(original) != "tt.unsplat":
            continue
        ids = source_ids(original.location)

        def refuse(message, *, ids=ids):
            raise ConversionError(
                message, opcode="tt.unsplat", source_id=next(iter(ids), None)
            )

        def fields(operation, allowed, *, refuse=refuse):
            if (
                operation.regions
                or operation.successors
                or any(
                    name not in {"op_name", "op_name__", *allowed}
                    and not name.startswith("prov.")
                    for name in (*operation.attributes, *operation.properties)
                )
            ):
                refuse("scalar tile read has an unqualified source property or region")

        fields(original, ())
        if len(original.operands) != 1 or len(original.results) != 1:
            refuse("scalar tile read requires one tensor and one scalar result")
        source = original.operands[0]
        if isinstance(source.type, builtin.TensorType) and not isinstance(
            source.type.element_type, (builtin.IntegerType, builtin.AnyFloat)
        ):
            # Logical pointer projections remain with the existing pointer
            # normalization; a numeric read grants no address interpretation.
            continue
        if (
            not numeric_tile(source.type)
            or prod(source.type.get_shape()) != 1
            or original.results[0].type != source.type.element_type
        ):
            refuse("scalar tile read requires one static unencoded numeric element")
        gather, splat, axis = None, None, None
        if hasattr(source.owner, "name") and opcode(source.owner) == "tt.splat":
            splat = source.owner
            fields(splat, ())
            if (
                len(splat.operands) != 1
                or len(splat.results) != 1
                or splat.operands[0].type != source.type.element_type
            ):
                refuse(
                    "singleton splat projection requires the complete scalar prototype"
                )
        if hasattr(source.owner, "name") and opcode(source.owner) == "tt.gather":
            gather = source.owner
            fields(gather, ("axis", "efficient_layout"))
            if len(gather.operands) != 2 or len(gather.results) != 1:
                refuse(
                    "scalar gather requires the complete source/index/result prototype"
                )
            table, indices = gather.operands
            axis = gather.properties.get("axis", gather.attributes.get("axis"))
            efficient = gather.properties.get(
                "efficient_layout", gather.attributes.get("efficient_layout")
            )
            if (
                not numeric_tile(table.type)
                or not numeric_tile(indices.type)
                or not isinstance(axis, builtin.IntegerAttr)
                or axis.type != builtin.i32
                or not 0 <= axis.value.data < len(source.type.get_shape())
                or len(table.type.get_shape()) != len(source.type.get_shape())
                or indices.type.get_shape() != source.type.get_shape()
                or indices.type.element_type not in (builtin.i32, builtin.i64)
                or table.type.element_type != source.type.element_type
                or any(
                    n != 1
                    for dim, n in enumerate(table.type.get_shape())
                    if dim != axis.value.data
                )
                or efficient is not None
                and not isinstance(efficient, builtin.UnitAttr)
            ):
                refuse(
                    "scalar gather requires a static single-axis numeric tile and one i32/i64 index"
                )
            axis = axis.value.data
            if hasattr(indices.owner, "name") and opcode(indices.owner) == "tt.splat":
                splat = indices.owner
                fields(splat, ())
                if (
                    len(splat.operands) != 1
                    or len(splat.results) != 1
                    or splat.operands[0].type != indices.type.element_type
                ):
                    refuse(
                        "scalar gather index splat differs from its original word type"
                    )
            # A shared tensor result keeps its ordinary tensor gather. Fuse
            # only a complete set of scalar projections, retaining one actual
            # lookup at the original gather, even across conditional users.
            if any(opcode(use.operation) != "tt.unsplat" for use in source.uses):
                gather, splat, axis = None, None, None
        plans.append((original, source, gather, splat, axis))

    reports, values, dead = [], {}, []
    for original, source, gather, splat, axis in plans:
        emitted = []
        related = [
            original,
            *([gather] if gather is not None else []),
            *([splat] if splat is not None else []),
        ]
        correspondence = list(
            dict.fromkeys(
                [
                    *related,
                    *(
                        [use.operation for use in source.uses]
                        if gather is not None
                        else []
                    ),
                ]
            )
        )
        location = builtin.FusedLoc(
            builtin.ArrayAttr(op.location for op in correspondence), builtin.NoneAttr()
        )

        def add(operation, *, emitted=emitted, location=location):
            operation.location = location
            emitted.append(operation)
            if generated_guards is not None and isinstance(operation, cf.AssertOp):
                generated_guards.add(operation)
            return operation.results[0] if operation.results else None

        if gather in values:
            value = values[gather]
        elif gather is None and splat is not None:
            value = splat.operands[0]
        elif gather is None:
            zero = add(arith.ConstantOp.from_int_and_width(0, builtin.IndexType()))
            value = add(
                tensor.ExtractOp(
                    source,
                    [zero] * len(source.type.get_shape()),
                    original.results[0].type,
                )
            )
        else:
            zero = add(arith.ConstantOp.from_int_and_width(0, builtin.IndexType()))
            table, indices = gather.operands
            word = (
                splat.operands[0]
                if splat is not None
                else add(
                    tensor.ExtractOp(
                        indices,
                        [zero] * len(indices.type.get_shape()),
                        indices.type.element_type,
                    )
                )
            )
            coordinate = guarded_axis_index(word, table.type.get_shape()[axis], add)
            value = add(
                tensor.ExtractOp(
                    table,
                    [
                        coordinate if dim == axis else zero
                        for dim in range(len(table.type.get_shape()))
                    ],
                    original.results[0].type,
                )
            )
        source_shape = list(
            (gather.operands[0] if gather is not None else source).type.get_shape()
        )
        efficient_layout = gather is not None and isinstance(
            gather.properties.get(
                "efficient_layout", gather.attributes.get("efficient_layout")
            ),
            builtin.UnitAttr,
        )
        if emitted:
            value.owner.attributes.update(
                {
                    "prov.family": builtin.StringAttr("tensor-read"),
                    "prov.op": builtin.StringAttr(
                        "lookup" if gather is not None else "unsplat"
                    ),
                    "prov.composite_family": builtin.StringAttr(
                        "tensor_lookup" if gather is not None else "tensor_projection"
                    ),
                    "prov.lookup_source_shape": builtin.DenseArrayBase.from_list(
                        builtin.i64, source_shape
                    ),
                }
            )
            if axis is not None:
                value.owner.attributes["prov.lookup_axis"] = builtin.IntegerAttr(
                    axis, builtin.i64
                )
            if efficient_layout:
                value.owner.attributes["prov.source_efficient_layout"] = (
                    builtin.UnitAttr()
                )
        if gather is not None:
            if emitted:
                gather.parent.insert_ops_before(emitted, gather)
                values[gather] = value
            Rewriter.replace_op(original, [], [value])
            dead.extend((gather, splat))
        else:
            Rewriter.replace_op(original, emitted, [value])
            if splat is not None:
                dead.append(splat)
        reports.append(
            {
                "kind": "source-scalar-tile-read",
                "source_ids": list(
                    dict.fromkeys(
                        identity
                        for op in related
                        for identity in source_ids(op.location)
                    )
                ),
                "tensor_source": "immutable-SSA-value; no-external-memory-stability-grant",
                "expansion": "guarded-single-axis-tensor-extract"
                if gather is not None
                else "singleton-splat-scalar-projection"
                if splat is not None
                else "singleton-tensor-extract",
                "axis": axis,
                "source_shape": source_shape,
                "source_efficient_layout_hint": efficient_layout,
                "control_position": "unchanged",
                "read_position": "original-gather"
                if gather is not None
                else "original-unsplat",
                "physical_mapping": "undecided",
            }
        )
    dead = list(dict.fromkeys(operation for operation in dead if operation is not None))
    for operation in [
        *[op for op in dead if opcode(op) != "tt.splat"],
        *[op for op in dead if opcode(op) == "tt.splat"],
    ]:
        if operation is not None and not any(
            result.uses for result in operation.results
        ):
            Rewriter.erase_op(operation)
    return reports
