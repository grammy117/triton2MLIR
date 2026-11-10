"""Logical MX unpack/scale tiles and an executable standard contraction.

References: Triton f797708c (DotScaledOp/verifier and test_matmul.py),
Ascend 8f0ade66 (DotScaledConverter/optional-scale tests), OCP MX v1.0 §6.
The GPU-layout fallback cannot run on pre-layout TTIR; Ascend's native MX
operation is not a standard payload. This adapter supplies that missing edge.
"""

from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, IndexOp, MatmulOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .standard_floats import _FP8_BY_NAME

FORMATS = {0: "e4m3", 1: "e5m2", 2: "e2m3", 3: "e3m2", 4: "e2m1", 5: "bf16", 6: "fp16"}
PROFILE = "logical-mx-f64-v1"
TYPES = {0: "f8E4M3FN", 1: "f8E5M2", 5: "bf16", 6: "f16"}


def contract(op, source_id=None):
    def refuse(message):
        raise ConversionError(message, opcode="tt.dot_scaled", source_id=source_id)

    attrs = {**op.attributes, **op.properties}
    allowed = {
        "op_name",
        "op_name__",
        "a_elem_type",
        "b_elem_type",
        "fastMath",
        "lhs_k_pack",
        "rhs_k_pack",
        "operandSegmentSizes",
    }
    if any(key not in allowed and not key.startswith("prov.") for key in attrs):
        refuse("scaled dot has an unqualified source property")
    segments = attrs.get("operandSegmentSizes")
    if (
        not isinstance(segments, builtin.DenseArrayBase)
        or segments.elt_type != builtin.i32
    ):
        refuse("scaled dot requires its typed operand segments")
    segments = tuple(segments.iter_values())
    if (
        len(segments) != 5
        or segments[:3] != (1, 1, 1)
        or any(n not in (0, 1) for n in segments[3:])
        or sum(segments) != len(op.operands)
        or len(op.results) != 1
    ):
        refuse("scaled dot operand segments do not describe A/B/C and optional scales")

    def integer(name, bits, default=None):
        value = attrs.get(name)
        if value is None and default is not None:
            return default
        if not isinstance(
            value, builtin.IntegerAttr
        ) or value.type != builtin.IntegerType(bits):
            refuse(f"scaled dot requires a typed {name} attribute")
        return value.value.data

    formats = [integer("a_elem_type", 32), integer("b_elem_type", 32)]
    if any(fmt not in FORMATS for fmt in formats):
        refuse("unknown pinned scaled-dot element format")
    if any(fmt in (2, 3) for fmt in formats):
        refuse(
            "reserved FP6 scaled-dot enum has no pinned Triton storage/packing contract"
        )
    packs = [bool(integer("lhs_k_pack", 1, 1)), bool(integer("rhs_k_pack", 1, 1))]
    fast = bool(integer("fastMath", 1))
    types = [value.type for value in (*op.operands, *op.results)]
    if any(
        not isinstance(typ, builtin.TensorType)
        or any(n <= 0 for n in typ.get_shape())
        or typ.encoding != builtin.NoneAttr()
        for typ in types
    ):
        refuse("scaled dot requires unencoded, positive static logical tiles")
    a, b, c = types[:3]
    rank = len(c.get_shape())
    if rank not in (2, 3) or any(len(typ.get_shape()) != rank for typ in types):
        refuse("scaled dot requires matching rank-two or batched rank-three tiles")
    if c != types[-1] or str(c.element_type) not in ("f16", "bf16", "f32", "f64"):
        refuse("scaled dot requires a matching standard floating accumulator/result")
    shapes = [list(a.get_shape()), list(b.get_shape())]
    for side, (fmt, pack, typ) in enumerate(zip(formats, packs, (a, b), strict=True)):
        if fmt == 4:
            if typ.element_type != builtin.i8:
                refuse("FP4 scaled dot requires packed i8 storage")
            shapes[side][-1 if (side == 0) == pack else -2] *= 2
        elif not pack:
            refuse("M/N packing is defined only for FP4")
        elif str(typ.element_type) != TYPES[fmt] and not (
            fmt in (0, 1) and typ.element_type == builtin.i8
        ):
            refuse("scaled-dot format disagrees with the source storage type")
    ashape, bshape = shapes
    batch, m, k, n = ashape[:-2], ashape[-2], ashape[-1], bshape[-1]
    if bshape[:-2] != batch or bshape[-2] != k or list(c.get_shape()) != [*batch, m, n]:
        refuse("scaled-dot logical batch/M/N/K dimensions disagree")
    groups, scales = [], []
    offset = 3
    for side, present in enumerate(segments[3:]):
        scale = op.operands[offset] if present else None
        offset += present
        group = None
        if scale is not None:
            dtype = str(scale.type.element_type)
            if dtype not in ("i8", "f8E4M3FN"):
                refuse("scaled-dot scales require E8M0 i8 or E4M3FN storage")
            group = 16 if dtype == "f8E4M3FN" else 32
            if k % group or list(scale.type.get_shape()) != [
                *batch,
                (m, n)[side],
                k // group,
            ]:
                refuse(
                    "scaled-dot scale shape disagrees with logical row/column and K groups"
                )
        scales.append(scale)
        groups.append(group)
    return {
        "profile": PROFILE,
        "element_formats": [FORMATS[n] for n in formats],
        "packing_axes": ["K" if p else ("M", "N")[i] for i, p in enumerate(packs)],
        "logical_shapes": shapes,
        "scale_groups": groups,
        "scale_formats": [
            str(s.type.element_type) if s is not None else None for s in scales
        ],
        "fast_math_permitted": fast,
        "nan_policy": "propagate; computed payload unspecified",
        "intermediate_type": "f64",
        "result_type": str(c.element_type),
        "accumulator": "source-C-extended-before-contraction",
        "physical_mapping": "undecided",
    }


def expand_scaled_dots(module, source_ids):
    from .shared_tensor import opcode

    # Validate the entire selection before mutating any source operation.
    plans = [
        (op, source_ids(op.location))
        for op in module.walk()
        if opcode(op) == "tt.dot_scaled"
    ]
    plans = [(op, ids, contract(op, next(iter(ids), None))) for op, ids in plans]
    reports = []
    for original, ids, info in plans:
        operations = []

        def append(op):
            op.location = original.location
            operations.append(op)
            return op.results[0]

        def pointwise(inputs, shape, maps, build, output_type=builtin.f64):
            result_type = builtin.TensorType(output_type, shape)
            empty = append(tensor.EmptyOp([], result_type))
            block = Block(
                arg_types=[*(v.type.element_type for v in inputs), output_type]
            )

            def emit(op):
                op.location = original.location
                block.add_op(op)
                return op.results[0] if op.results else None

            emit(YieldOp(build(emit, block.args)))
            return append(
                GenericOp(
                    inputs,
                    [empty],
                    Region(block),
                    [
                        builtin.AffineMapAttr(m)
                        for m in (*maps, AffineMap.identity(len(shape)))
                    ],
                    [IteratorTypeAttr.parallel()] * len(shape),
                    [result_type],
                )
            )

        def extend(emit, value):
            if value.type == builtin.bf16:
                value = emit(arith.ExtFOp(value, builtin.f32))
            return (
                value
                if value.type == builtin.f64
                else emit(arith.ExtFOp(value, builtin.f64))
            )

        def iconst(emit, number):
            return emit(arith.ConstantOp.from_int_and_width(number, builtin.i64))

        def fconst(emit, number):
            return emit(arith.ConstantOp(builtin.FloatAttr(number, builtin.f64)))

        offset, operands = 3, []
        for side in range(2):
            source = original.operands[side]
            fmt, shape = info["element_formats"][side], info["logical_shapes"][side]
            rank = len(shape)
            dims = [AffineDimExpr(i) for i in range(rank)]
            packed_axis = rank - (
                1 if (side == 0) == (info["packing_axes"][side] == "K") else 2
            )
            data_map = list(dims)
            if fmt == "e2m1":
                data_map[packed_axis] = data_map[packed_axis] // 2
            inputs, maps = [source], [AffineMap(rank, 0, tuple(data_map))]
            group = info["scale_groups"][side]
            scale = original.operands[offset] if group is not None else None
            offset += scale is not None
            if scale is not None:
                scale_map = (
                    *dims[:-2],
                    dims[-2 if side == 0 else -1],
                    dims[-1 if side == 0 else -2] // group,
                )
                inputs.append(scale)
                maps.append(AffineMap(rank, 0, scale_map))

            def decode(emit, args):
                value = args[0]
                if fmt == "e2m1":
                    word = emit(arith.ExtUIOp(value, builtin.i64))
                    coordinate = emit(
                        arith.IndexCastOp(emit(IndexOp(packed_axis)), builtin.i64)
                    )
                    parity = emit(arith.AndIOp(coordinate, iconst(emit, 1)))
                    shift = emit(arith.MuliOp(parity, iconst(emit, 4)))
                    nibble = emit(arith.ShRUIOp(word, shift))
                    exponent = emit(
                        arith.AndIOp(
                            emit(arith.ShRUIOp(nibble, iconst(emit, 1))),
                            iconst(emit, 3),
                        )
                    )
                    mantissa = emit(arith.AndIOp(nibble, iconst(emit, 1)))
                    subnormal = emit(arith.CmpiOp(exponent, iconst(emit, 0), "eq"))
                    leading = emit(
                        arith.SelectOp(subnormal, iconst(emit, 0), iconst(emit, 2))
                    )
                    significand = emit(
                        arith.UIToFPOp(
                            emit(arith.AddiOp(mantissa, leading)), builtin.f64
                        )
                    )
                    exponent = emit(
                        arith.SelectOp(subnormal, iconst(emit, 1), exponent)
                    )
                    power = emit(arith.AddiOp(exponent, iconst(emit, 1021)))
                    bits = emit(arith.ShLIOp(power, iconst(emit, 52)))
                    magnitude = emit(
                        arith.MulfOp(
                            significand, emit(arith.BitcastOp(bits, builtin.f64))
                        )
                    )
                    sign = emit(arith.AndIOp(nibble, iconst(emit, 8)))
                    sign = emit(arith.ShLIOp(sign, iconst(emit, 60)))
                    value = emit(
                        arith.BitcastOp(
                            emit(
                                arith.OrIOp(
                                    emit(arith.BitcastOp(magnitude, builtin.i64)), sign
                                )
                            ),
                            builtin.f64,
                        )
                    )
                else:
                    if value.type == builtin.i8:
                        target = _FP8_BY_NAME[
                            {"e4m3": "f8E4M3FN", "e5m2": "f8E5M2"}[fmt]
                        ]
                        value = emit(arith.BitcastOp(value, target))
                    value = extend(emit, value)
                if scale is not None:
                    word = args[1]
                    if word.type == builtin.i8:
                        word = emit(arith.ExtUIOp(word, builtin.i64))
                        # All finite E8M0 values, including code zero, are normal
                        # in f64. No exp2 approximation or premature f32 overflow.
                        power = emit(arith.AddiOp(word, iconst(emit, 896)))
                        factor = emit(
                            arith.BitcastOp(
                                emit(arith.ShLIOp(power, iconst(emit, 52))), builtin.f64
                            )
                        )
                        nan = emit(arith.CmpiOp(word, iconst(emit, 255), "eq"))
                        factor = emit(
                            arith.SelectOp(nan, fconst(emit, float("nan")), factor)
                        )
                    else:
                        factor = extend(emit, word)
                    value = emit(arith.MulfOp(value, factor))
                return value

            operands.append(pointwise(inputs, shape, maps, decode))
        c = original.operands[2]
        cshape = c.type.get_shape()
        initial = pointwise(
            [c],
            cshape,
            [AffineMap.identity(len(cshape))],
            lambda emit, args: extend(emit, args[0]),
        )
        if len(cshape) == 2:
            result = append(MatmulOp(operands, [initial], res=[initial.type]))
        else:
            body = Block(arg_types=[builtin.f64] * 3)
            multiply = arith.MulfOp(body.args[0], body.args[1])
            add = arith.AddfOp(multiply, body.args[2])
            body.add_ops([multiply, add, YieldOp(add)])
            for child in body.ops:
                child.location = original.location
            maps = [(0, 1, 3), (0, 3, 2), (0, 1, 2)]
            result = append(
                GenericOp(
                    operands,
                    [initial],
                    Region(body),
                    [
                        builtin.AffineMapAttr(
                            AffineMap(4, 0, tuple(AffineDimExpr(i) for i in mapping))
                        )
                        for mapping in maps
                    ],
                    [IteratorTypeAttr.parallel()] * 3 + [IteratorTypeAttr.reduction()],
                    [initial.type],
                )
            )
        if c.type.element_type != builtin.f64:
            result = pointwise(
                [result],
                cshape,
                [AffineMap.identity(len(cshape))],
                lambda emit, args: emit(arith.TruncFOp(args[0], c.type.element_type)),
                c.type.element_type,
            )
        Rewriter.replace_op(original, operations, [result])
        reports.append({"kind": "source-scaled-dot", "source_ids": ids, **info})
    return reports
