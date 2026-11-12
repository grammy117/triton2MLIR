"""Executable IEEE f16/BF16/f32/f64 and FTZ f32 atomic addition.

Meta 17f0c234's unstructured atomic conversion supplies the typed parallel
event, mask and old value. This adapter supplies the missing numeric policy
inside standard scalar arithmetic and a standard generic atomic region.
"""

from xdsl.dialects import arith, builtin, memref
from xdsl.ir import Block, Region
from xdsl.rewriter import Rewriter

from .diagnostics import ConversionError
from .standard_memref import AtomicYieldOp, GenericAtomicRMWOp

CONTRACT = "f32-rne-sign-preserving-ftz-add"
PAYLOAD = "standard-memref-generic-f32-ftz-add-with-original-returned-old-value"
IEEE_PAYLOAD = "standard-memref-generic-f32-quiet-add-with-original-returned-old-value"
F64_CONTRACT = "ieee-f64-rne-add-quiet-nans"
F64_PAYLOAD = "standard-memref-generic-f64-quiet-add-with-original-returned-old-value"
HALF_CONTRACTS = {
    "f16": "ieee-f16-rne-add-quiet-nans",
    "bf16": "ieee-bf16-rne-add-quiet-nans",
}
HALF_PAYLOADS = {
    name: f"standard-memref-generic-{name}-quiet-add-with-original-returned-old-value"
    for name in HALF_CONTRACTS
}


def flush(value, *, quiet=False, ftz=True):
    """Zero exponent means zero/subnormal; retaining only sign preserves zero signs."""
    element = value.type
    formats = {
        builtin.f16: (builtin.i16, 0x7C00, 10),
        builtin.bf16: (builtin.i16, 0x7F80, 7),
        builtin.f32: (builtin.i32, 0x7F800000, 23),
        builtin.f64: (builtin.i64, 0x7FF0000000000000, 52),
    }
    if element not in formats or ftz and element != builtin.f32:
        raise ConversionError(
            "floating atomic regions require a qualified IEEE type or FTZ f32"
        )
    integer, exponent_bits, fraction_bits = formats[element]
    bits = arith.BitcastOp(value, integer)
    exponent = arith.ConstantOp.from_int_and_width(exponent_bits, integer)
    zero = arith.ConstantOp.from_int_and_width(0, integer)
    operations, word = [bits, exponent, zero], bits.result
    if ftz:
        sign = arith.ConstantOp.from_int_and_width(-0x80000000, builtin.i32)
        field = arith.AndIOp(bits.result, exponent.result)
        small = arith.CmpiOp(field.result, zero.result, "eq")
        signed_zero = arith.AndIOp(bits.result, sign.result)
        selected = arith.SelectOp(small.result, signed_zero.result, bits.result)
        operations.extend([sign, field, small, signed_zero, selected])
        word = selected.result
    if quiet:
        fraction = arith.ConstantOp.from_int_and_width(
            (1 << fraction_bits) - 1, integer
        )
        quiet_bit = arith.ConstantOp.from_int_and_width(
            1 << (fraction_bits - 1), integer
        )
        word_field = arith.AndIOp(word, exponent.result)
        all_exponent = arith.CmpiOp(word_field.result, exponent.result, "eq")
        mantissa = arith.AndIOp(word, fraction.result)
        nonzero = arith.CmpiOp(mantissa.result, zero.result, "ne")
        nan = arith.AndIOp(all_exponent.result, nonzero.result)
        quiet_word = arith.OrIOp(word, quiet_bit.result)
        quieted = arith.SelectOp(nan.result, quiet_word.result, word)
        operations.extend(
            [
                fraction,
                quiet_bit,
                word_field,
                all_exponent,
                mantissa,
                nonzero,
                nan,
                quiet_word,
                quieted,
            ]
        )
        word = quieted.result
    result = arith.BitcastOp(word, element)
    return [*operations, result], result.result


def materialize(op, *, ftz=True):
    """Refine the admitted upstream typed add event, retaining its placement."""
    if (
        not isinstance(op, memref.AtomicRMWOp)
        or op.kind.value.data != 0
        or op.results[0].type
        not in (builtin.f16, builtin.bf16, builtin.f32, builtin.f64)
        or op.operands[0].type != op.results[0].type
        or ftz
        and op.results[0].type != builtin.f32
    ):
        raise ConversionError(
            "floating addition requires the original standard typed atomic add"
        )
    outside, value = flush(op.operands[0]) if ftz else ([], op.operands[0])
    body = Block(arg_types=[op.results[0].type])
    current_ops, current = flush(body.args[0]) if ftz else ([], body.args[0])
    arithmetic = []
    if current.type == builtin.bf16:
        lhs, rhs = arith.ExtFOp(current, builtin.f32), arith.ExtFOp(value, builtin.f32)
        summed = arith.AddfOp(lhs.result, rhs.result)
        updated = arith.TruncFOp(summed.result, builtin.bf16)
        arithmetic.extend((lhs, rhs, summed))
    else:
        updated = arith.AddfOp(current, value)
    result_ops, result = flush(updated.result, quiet=True, ftz=ftz)
    body.add_ops(
        [
            *current_ops,
            *arithmetic,
            updated,
            *result_ops,
            AtomicYieldOp.create(operands=[result]),
        ]
    )
    atomic = GenericAtomicRMWOp.create(
        operands=[*op.operands[1:]],
        result_types=[op.results[0].type],
        regions=[Region(body)],
    )
    for outer in [*outside, atomic]:
        for nested in outer.walk():
            nested.location = op.location
    Rewriter.replace_op(op, [*outside, atomic], atomic.results)
    return atomic
