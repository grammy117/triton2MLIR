"""Executable FP8 conversion policies above backend-specific cast selection.

The pinned Triton backends differ on saturation and infinity conversion.
Require that source contract explicitly, then retain its clamps, selections
and rounding in standard MLIR. Native standard RTNE truncation is a separate
consumer implementation; provenance supplies no numeric permission.
"""

from xdsl.dialects import arith, builtin, math, tensor
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .standard_floats import FP8_TYPE_NAMES

POLICIES = ("ieee", "satfinite", "satfinite-nan-inf")
PER_CAST = "per-cast"
SELECTIONS = (*POLICIES, PER_CAST)
SOURCE_POLICY = "t2m.fp8_overflow"
PROFILE = "t2m.per-cast-fp8.v1"
FORMATS = {
    "f8E4M3FN": (448.0, 127),
    "f8E4M3FNUZ": (240.0, 128),
    "f8E5M2": (57344.0, 126),
    "f8E5M2FNUZ": (57344.0, 128),
}
STANDARD = {"f16", "bf16", "f32", "f64"}


def _plans(module, source_ids, overflow):
    from .shared_tensor import opcode

    if overflow not in (None, *SELECTIONS):
        raise ValueError(
            "FP8 overflow policy must be ieee, satfinite, satfinite-nan-inf or per-cast"
        )
    plans = []
    for op in module.walk():
        name = opcode(op)
        semantic = op.attributes.get(SOURCE_POLICY, op.properties.get(SOURCE_POLICY))
        if semantic is not None and (overflow != PER_CAST or name != "tt.fp_to_fp"):
            raise ConversionError("source FP8 policy requires the explicit per-cast profile on tt.fp_to_fp",
                opcode=name, source_id=next(iter(source_ids(op.location)), None))
        if semantic is not None and (len(op.results) != 1 or SOURCE_POLICY in op.attributes and SOURCE_POLICY in op.properties):
            raise ConversionError("source FP8 policy requires one result and one unambiguous attribute", opcode=name)
        # Standard arith.truncf already has its own numeric contract. A TTIR
        # profile must never reinterpret it as a saturating source cast.
        if name != "tt.fp_to_fp" or len(op.results) != 1:
            continue
        output = op.results[0].type
        target = (
            output.element_type if isinstance(output, builtin.TensorType) else output
        )
        dst = str(target)
        if dst not in FORMATS:
            if semantic is not None:
                raise ConversionError("source FP8 policy is attached to a non-FP8 conversion", opcode=name)
            continue
        ids = source_ids(op.location)

        def refuse(message):
            raise ConversionError(message, opcode=name, source_id=next(iter(ids), None))

        if len(op.operands) != 1:
            refuse("FP8 conversion requires one input and result")
        source_type = op.operands[0].type
        source = (
            source_type.element_type
            if isinstance(source_type, builtin.TensorType)
            else source_type
        )
        if source == target:
            if semantic is not None:
                refuse("identity conversions cannot acquire overflow or infinity semantics")
            continue
        if str(source) not in STANDARD | set(FP8_TYPE_NAMES):
            refuse("FP8 conversion requires a declared standard floating format")
        if isinstance(source_type, builtin.TensorType) != isinstance(
            output, builtin.TensorType
        ):
            refuse("FP8 conversion must preserve scalar/tensor shape")
        if isinstance(output, builtin.TensorType) and (
            source_type.get_shape() != output.get_shape()
            or any(n < 0 for n in output.get_shape())
        ):
            refuse("FP8 source tiles require matching static shapes")
        selected = semantic.data if isinstance(semantic, builtin.StringAttr) else None
        if overflow == PER_CAST and selected not in POLICIES:
            refuse("each source FP8 conversion needs its own explicit valid numeric policy")
        if overflow != PER_CAST:
            selected = overflow
        if selected is None:
            refuse("FP8 conversion requires an explicit overflow/infinity policy")
        if any(
            key not in ("op_name", "op_name__", "rounding", SOURCE_POLICY)
            and not key.startswith("prov.")
            for key in (*op.attributes, *op.properties)
        ):
            refuse("FP8 conversion has an unqualified source property")
        rounding = op.properties.get("rounding", op.attributes.get("rounding"))
        if rounding is None and str(source) in FP8_TYPE_NAMES:
            mode = "rtne"
        elif (
            isinstance(rounding, builtin.IntegerAttr)
            and rounding.type == builtin.i32
            and rounding.value.data in (0, 1)
        ):
            mode = "rtz" if rounding.value.data == 0 else "rtne"
        else:
            refuse("FP8 narrowing requires the pinned explicit RTNE/RTZ mode")
        plans.append((op, source, target, mode, ids, selected))
    return plans


def validate_fp8_casts(module, source_ids, *, overflow=None):
    """Validate every source choice before inlining/folding can discard it."""
    _plans(module, source_ids, overflow)


def expand_fp8_casts(module, source_ids, *, overflow=None, tensor_carrier="linalg"):
    if tensor_carrier not in ("linalg", "arith"):
        raise ValueError("FP8 conversion requires a standard tensor computation carrier")
    plans = _plans(module, source_ids, overflow)
    reports = []
    per_cast = overflow == PER_CAST
    for original, source, target, mode, ids, selected in plans:
        overflow = selected
        operations = []

        def emit(op):
            op.location = original.location
            operations.append(op)
            return op.results[0]

        shaped = isinstance(original.results[0].type, builtin.TensorType)
        block = Block(arg_types=[source, target]) if shaped else None
        value = block.args[0] if shaped else original.operands[0]
        compute_type = source
        if str(source) in (*FP8_TYPE_NAMES, "f16", "bf16"):
            compute_type = builtin.f32
            value = emit(arith.ExtFOp(value, compute_type))
        maximum, nan_word = FORMATS[str(target)]
        needs_clamp = overflow != "ieee" or mode == "rtz"
        magnitude = emit(math.AbsFOp(value))
        inf = emit(arith.ConstantOp(builtin.FloatAttr(float("inf"), compute_type)))
        is_inf = emit(arith.CmpfOp(magnitude, inf, "oeq"))
        adjusted = value
        if needs_clamp:
            low = emit(arith.ConstantOp(builtin.FloatAttr(-maximum, compute_type)))
            high = emit(arith.ConstantOp(builtin.FloatAttr(maximum, compute_type)))
            clipped = emit(arith.MaximumfOp(value, low))
            clipped = emit(arith.MinimumfOp(clipped, high))
            adjusted = (
                emit(arith.SelectOp(is_inf, value, clipped))
                if overflow == "ieee"
                else clipped
            )
        narrowed = emit(arith.TruncFOp(adjusted, target))
        if mode == "rtz":
            back = emit(arith.ExtFOp(narrowed, compute_type))
            back_abs = emit(math.AbsFOp(back))
            overshot = emit(arith.CmpfOp(back_abs, magnitude, "ogt"))
            bits = emit(arith.BitcastOp(narrowed, builtin.i8))
            one = emit(arith.ConstantOp.from_int_and_width(1, builtin.i8))
            lower = emit(arith.SubiOp(bits, one))
            if str(target).endswith("FNUZ"):
                negzero = emit(arith.ConstantOp.from_int_and_width(128, builtin.i8))
                zero = emit(arith.ConstantOp.from_int_and_width(0, builtin.i8))
                is_negzero = emit(arith.CmpiOp(lower, negzero, "eq"))
                lower = emit(arith.SelectOp(is_negzero, zero, lower))
            bits = emit(arith.SelectOp(overshot, lower, bits))
            narrowed = emit(arith.BitcastOp(bits, target))
        if overflow == "satfinite-nan-inf":
            bits = emit(arith.ConstantOp.from_int_and_width(nan_word, builtin.i8))
            nan = emit(arith.BitcastOp(bits, target))
            narrowed = emit(arith.SelectOp(is_inf, nan, narrowed))
        if shaped and tensor_carrier == "arith":
            # Native TTIR normalizers load standard arith/math, but not Linalg.
            # Tensorize the same scalar recipe before any of those passes.
            # The later shared arithmetic conversion supplies parallel Linalg.
            output = original.results[0].type
            shape = output.get_shape()
            mapper = {block.args[0]: original.operands[0]}
            tensor_operations = []
            for operation in operations:
                type_ = builtin.TensorType(operation.results[0].type, shape)
                properties = dict(operation.properties)
                if isinstance(operation, arith.ConstantOp):
                    value = properties["value"]
                    properties["value"] = builtin.DenseIntOrFPElementsAttr.from_list(type_, [value.value.data])
                converted = type(operation).create(operands=[mapper[value] for value in operation.operands],
                    result_types=[type_], attributes=dict(operation.attributes), properties=properties,
                    location=operation.location)
                mapper[operation.results[0]] = converted.results[0]
                tensor_operations.append(converted)
            narrowed = mapper[narrowed]
            operations = tensor_operations
            root = narrowed.owner
        elif shaped:
            terminal = YieldOp(narrowed)
            terminal.location = original.location
            block.add_ops([*operations, terminal])
            output = original.results[0].type
            shape = output.get_shape()
            empty = tensor.EmptyOp([], output)
            root = GenericOp(
                [original.operands[0]],
                [empty.tensor],
                Region(block),
                [builtin.AffineMapAttr(AffineMap.identity(len(shape)))] * 2,
                [IteratorTypeAttr.parallel()] * len(shape),
                [output],
            )
            operations = [empty, root]
            narrowed = root.results[0]
        else:
            root = narrowed.owner
        root.attributes.update(
            {
                key: value
                for key, value in (
                    *original.attributes.items(),
                    *original.properties.items(),
                )
                if key.startswith("prov.")
            }
        )
        root.attributes.update(
            {
                "prov.composite_family": builtin.StringAttr("fp8_conversion"),
                "prov.cast_source": builtin.StringAttr(str(source)),
                "prov.cast_target": builtin.StringAttr(str(target)),
                "prov.cast_rounding": builtin.StringAttr(mode),
                "prov.fp8_overflow": builtin.StringAttr(overflow),
            }
        )
        for op in operations:
            op.location = original.location
        Rewriter.replace_op(original, operations, [narrowed])
        reports.append(
            {
                "kind": "source-fp8-conversion",
                "source_ids": ids,
                "source_type": str(source),
                "target_type": str(target),
                "rounding": mode,
                "overflow_policy": overflow,
                "finite_overflow": "standard" if overflow == "ieee" else "saturate",
                "infinity": "standard"
                if overflow == "ieee"
                else "nan"
                if overflow == "satfinite-nan-inf"
                else "saturate",
                "unsigned_zero": str(target).endswith("FNUZ"),
                "computed_nan_payload": "unspecified",
                "expansion": "standard-clamps-casts-and-exact-RTZ-correction",
                "physical_mapping": "undecided",
                **({"tensor_carrier": "standard-elementwise-arith"} if shaped and tensor_carrier == "arith" else {}),
                **({"source_policy_profile": PROFILE, "source_policy_attribute": SOURCE_POLICY} if per_cast else {}),
            }
        )
    return reports
