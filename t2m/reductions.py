"""Retain general TTIR combiners as first-element-seeded standard reductions.

Meta/shared's ReduceConverter supports a selected single-operation roster and
supplies neutral constants; its special argmin/max converters cover selected
tuple forms. Reuse standard Linalg reduction semantics for the complete scalar
body instead. The first source element is the real initial value, including
NaNs, signed zeros and every tuple component. No mathematical identity is
invented and no lane/warp reduction algorithm or widened accumulator is chosen.
"""

from xdsl.dialects import builtin, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, ReduceOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .scans import _scalar_combiner


def _slice(source, offsets, sizes, result_type):
    return tensor.ExtractSliceOp.build(
        operands=[source, [], [], []],
        result_types=[result_type],
        properties={
            "static_offsets": builtin.DenseArrayBase.from_list(builtin.i64, offsets),
            "static_sizes": builtin.DenseArrayBase.from_list(builtin.i64, sizes),
            "static_strides": builtin.DenseArrayBase.from_list(
                builtin.i64, [1] * len(sizes)
            ),
        },
    )


def expand_reductions(module, source_ids, *, generated_guards=()):
    from .shared_tensor import opcode

    plans = []
    for original in list(module.walk()):
        if opcode(original) != "tt.reduce":
            continue
        ids = source_ids(original.location)

        def refuse(message, opcode="tt.reduce", *, ids=ids):
            raise ConversionError(
                message, opcode=opcode, source_id=next(iter(ids), None)
            )

        types = [value.type for value in original.operands]
        if (
            not types
            or len(types) != len(original.results)
            or not all(isinstance(type_, builtin.TensorType) for type_ in types)
        ):
            refuse(
                "reduction needs matching ranked tensor operands and complete results"
            )
        shape = types[0].get_shape()
        axis = original.properties.get("axis")
        if (
            not isinstance(axis, builtin.IntegerAttr)
            or not 0 <= axis.value.data < len(shape)
            or any(n <= 0 for n in shape)
            or any(type_.get_shape() != shape for type_ in types)
            or any(not isinstance(type_.encoding, builtin.NoneAttr) for type_ in types)
        ):
            refuse(
                "reduction requires equal positive unencoded source tiles and a valid axis"
            )
        axis = axis.value.data
        elements = [type_.element_type for type_ in types]
        reduced_shape = [extent for dim, extent in enumerate(shape) if dim != axis]
        results = [builtin.TensorType(element, reduced_shape) for element in elements]
        if not all(
            isinstance(element, (builtin.IntegerType, builtin.AnyFloat))
            for element in elements
        ) or any(
            value.type != (result if reduced_shape else element)
            for value, result, element in zip(
                original.results, results, elements, strict=True
            )
        ):
            refuse(
                "reduction result types must remove exactly the source axis and retain tuple formats"
            )
        if len(original.regions) != 1 or len(original.regions[0].blocks) != 1:
            refuse("reduction needs one inlined scalar combiner block")
        combiner = original.regions[0].block
        end = combiner.last_op
        if (
            [value.type for value in combiner.args] != elements * 2
            or end is None
            or opcode(end) != "tt.reduce.return"
            or [value.type for value in end.operands] != elements
        ):
            refuse(
                "reduction combiner must consume and return every original tuple component"
            )
        controls, body_ids = _scalar_combiner(
            combiner,
            end,
            refuse,
            source_ids,
            family="reduction",
            generated_guards=generated_guards,
        )
        plans.append(
            (
                original,
                ids,
                types,
                shape,
                axis,
                elements,
                results,
                combiner,
                end,
                controls,
                body_ids,
            )
        )

    reports = []
    # Validate all original candidates, including singleton/dead bodies, first.
    for (
        original,
        ids,
        types,
        shape,
        axis,
        elements,
        results,
        combiner,
        end,
        controls,
        body_ids,
    ) in plans:
        offsets, sizes = [0] * len(shape), list(shape)
        sizes[axis] = 1
        initial = [
            _slice(value, offsets, sizes, result_type)
            for value, result_type in zip(original.operands, results, strict=True)
        ]
        emitted = [*initial]
        values = [value.result for value in initial]
        if shape[axis] > 1:
            offsets, sizes = [0] * len(shape), list(shape)
            offsets[axis], sizes[axis] = 1, shape[axis] - 1
            remaining = [
                _slice(
                    value, offsets, sizes, builtin.TensorType(type_.element_type, sizes)
                )
                for value, type_ in zip(original.operands, types, strict=True)
            ]
            body = Block(arg_types=elements * 2)
            count = len(elements)
            # Linalg presents current input before accumulator; the TTIR
            # combiner's left/right operands retain their original order.
            mapping = {
                value: body.args[(index + count) % (2 * count)]
                for index, value in enumerate(combiner.args)
            }
            for child in list(combiner.ops)[:-1]:
                body.add_op(child.clone(value_mapper=mapping))
            terminal = YieldOp(*(mapping.get(value, value) for value in end.operands))
            terminal.location = end.location
            body.add_op(terminal)
            if count == 1:
                reduction = ReduceOp(
                    remaining[0].result,
                    initial[0].result,
                    builtin.DenseArrayBase.from_list(builtin.i64, [axis]),
                    Region(body),
                )
            else:
                output_map = AffineMap(
                    len(shape),
                    0,
                    tuple(
                        AffineDimExpr(index)
                        for index in range(len(shape))
                        if index != axis
                    ),
                )
                reduction = GenericOp(
                    [value.result for value in remaining],
                    values,
                    Region(body),
                    [builtin.AffineMapAttr(AffineMap.identity(len(shape)))] * count
                    + [builtin.AffineMapAttr(output_map)] * count,
                    [
                        IteratorTypeAttr.reduction()
                        if index == axis
                        else IteratorTypeAttr.parallel()
                        for index in range(len(shape))
                    ],
                    results,
                )
            reduction.attributes.update(
                {
                    "prov.composite_family": builtin.StringAttr("reduction"),
                    "prov.composite_id": builtin.StringAttr(
                        "reduction:" + (ids[0] if ids else str(original.location))
                    ),
                    "prov.reduction_initialization": builtin.StringAttr(
                        "first-original-element"
                    ),
                    "prov.reduction_axis": builtin.IntegerAttr(axis, builtin.i64),
                }
            )
            emitted.extend((*remaining, reduction))
            values = list(reduction.results)
        if len(shape) == 1:
            extracted = [
                tensor.ExtractOp(value, [], value.type.element_type) for value in values
            ]
            emitted.extend(extracted)
            values = [value.result for value in extracted]
        for child in emitted:
            child.location = original.location
            child.attributes.setdefault("prov.family", builtin.StringAttr("reduction"))
            child.attributes.setdefault("prov.op", builtin.StringAttr("reduce"))
            child.attributes.setdefault(
                "prov.composite_family", builtin.StringAttr("reduction")
            )
            child.attributes.setdefault(
                "prov.reduction_initialization",
                builtin.StringAttr("first-original-element"),
            )
        Rewriter.replace_op(original, emitted, values)
        reports.append(
            {
                "kind": "ttir-first-element-reduction",
                "source_ids": ids,
                "axis": axis,
                "shape": list(shape),
                "tuple_types": [str(element) for element in elements],
                "initialization": "first-original-element; no-invented-neutral-value",
                "combiner_argument_order": "original-left-accumulator; original-right-current-input",
                "combiner_control_flow": controls,
                "combiner_source_ids": body_ids,
                "expansion": "singleton-projection"
                if shape[axis] == 1
                else "standard-linalg-reduction",
                "physical_mapping": "undecided",
                "source_associativity": "source-contract; no-provenance-grant",
            }
        )
    return reports
