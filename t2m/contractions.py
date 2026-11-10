"""Source dot contracts retained beside ordinary typed linalg contractions.

Pinned TritonOps.td specifies that inputPrecision applies only to f32 input
operands. Preserve ignored settings rather than refusing other floating and
integer formats. Accuracy permissions and caller provenance do not replace
the accumulator or grant reassociation/approximation by themselves.
"""

import math

from xdsl.dialects import builtin

from .diagnostics import ConversionError

PRECISIONS = {0: "tf32", 1: "tf32x3", 2: "ieee", 3: "bf16x3", 4: "bf16x6"}


def zero_accumulator(value):
    """Match upstream's constant/splat zero cases while retaining its sign."""
    from .shared_tensor import opcode

    def producer(value):
        op = value.owner
        if hasattr(op, "name") and opcode(op) == "tt.splat" and len(op.operands) == 1:
            op = op.operands[0].owner
        return op

    op = producer(value)
    bits = None
    target = (
        value.type.element_type
        if isinstance(value.type, builtin.TensorType)
        else value.type
    )
    if (
        hasattr(op, "name")
        and opcode(op) in ("tt.bitcast", "arith.bitcast")
        and len(op.operands) == 1
    ):
        widths = {
            "f16": 16,
            "bf16": 16,
            "f32": 32,
            "f64": 64,
            "f8E4M3FN": 8,
            "f8E5M2": 8,
            "f8E4M3FNUZ": 8,
            "f8E5M2FNUZ": 8,
        }
        source = op.operands[0].type
        source = (
            source.element_type if isinstance(source, builtin.TensorType) else source
        )
        bits = widths.get(str(target))
        if not isinstance(source, builtin.IntegerType) or source.width.data != bits:
            return None
        op = producer(op.operands[0])
    if not hasattr(op, "name") or opcode(op) != "arith.constant":
        return None
    attr = op.properties.get("value", op.attributes.get("value"))
    if isinstance(attr, builtin.DenseIntOrFPElementsAttr):
        if not attr.is_splat():
            return None
        number = attr.get_values()[0]
    elif isinstance(attr, (builtin.FloatAttr, builtin.IntegerAttr)):
        number = attr.value.data
    else:
        return None
    if bits is not None:
        if not isinstance(number, int):
            return None
        word = number & ((1 << bits) - 1)
        if word == 0:
            return "positive"
        return (
            "negative"
            if word == 1 << (bits - 1) and not str(target).endswith("FNUZ")
            else None
        )
    if number != 0:
        return None
    return (
        "negative"
        if isinstance(number, float) and math.copysign(1, number) < 0
        else "positive"
    )


def dot_contract(op, source_id, *, allow_composite=False):
    def refuse(message):
        raise ConversionError(message, opcode="tt.dot", source_id=source_id)

    if len(op.operands) != 3 or len(op.results) != 1:
        refuse("dot requires both inputs and its original accumulator")
    types = [value.type for value in (*op.operands, *op.results)]
    if any(not isinstance(type_, builtin.TensorType) for type_ in types):
        refuse("dot requires ranked tensor operands and result")
    elements = [str(type_.element_type) for type_ in types]
    precision = op.properties.get("inputPrecision", op.attributes.get("inputPrecision"))
    if precision is None:
        mode = 2  # The pinned TTIR definition's explicit default is IEEE.
    elif (
        not isinstance(precision, builtin.IntegerAttr)
        or precision.type != builtin.i32
        or precision.value.data not in PRECISIONS
    ):
        refuse("dot has an unqualified input precision")
    else:
        mode = precision.value.data
    applies = "f32" in elements[:2]
    if applies and mode != 2 and not allow_composite:
        refuse(
            "f32 dot precision must explicitly be IEEE until its selected precision is expanded"
        )
    if applies and mode != 2 and (elements[:2] != ["f32", "f32"] or elements[2:] != ["f32", "f32"]):
        refuse("selected f32 dot precision requires f32 inputs, accumulator and result")
    bound = op.properties.get(
        "maxNumImpreciseAcc", op.attributes.get("maxNumImpreciseAcc")
    )
    if bound is not None and (
        not isinstance(bound, builtin.IntegerAttr)
        or bound.type != builtin.i32
        or bound.value.data < 0
    ):
        refuse("dot requires a nonnegative i32 imprecise accumulation bound")
    return {
        "input_formats": elements[:2],
        "accumulator_format": elements[2],
        "result_format": elements[3],
        "input_precision": PRECISIONS[mode],
        "input_precision_applies": applies,
        "max_imprecise_accumulations": 0 if bound is None else bound.value.data,
        "portable_accumulator": ("source-C-added-after-official-composite" if applies and mode in (1, 3, 4)
                                 else "original-source-C"),
        "physical_mapping": "undecided",
    }


def needs_dot_precision_expansion(op):
    from .shared_tensor import opcode
    if opcode(op) != "tt.dot" or len(op.operands) != 3:
        return False
    precision = op.properties.get("inputPrecision", op.attributes.get("inputPrecision"))
    return (isinstance(precision, builtin.IntegerAttr) and precision.type == builtin.i32
            and precision.value.data in (0, 1, 3, 4)
            and all(isinstance(value.type, builtin.TensorType) and value.type.element_type == builtin.f32
                    for value in op.operands[:2]))
