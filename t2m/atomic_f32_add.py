"""Executable IEEE/FTZ source f32 addition with explicit result NaN quieting.

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


def flush(value, *, quiet=False, ftz=True):
    """Zero exponent means zero/subnormal; retaining only sign preserves zero signs."""
    bits = arith.BitcastOp(value, builtin.i32)
    exponent = arith.ConstantOp.from_int_and_width(0x7f800000, builtin.i32)
    zero = arith.ConstantOp.from_int_and_width(0, builtin.i32)
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
        fraction = arith.ConstantOp.from_int_and_width(0x007fffff, builtin.i32)
        quiet_bit = arith.ConstantOp.from_int_and_width(0x00400000, builtin.i32)
        word_field = arith.AndIOp(word, exponent.result)
        all_exponent = arith.CmpiOp(word_field.result, exponent.result, "eq")
        mantissa = arith.AndIOp(word, fraction.result)
        nonzero = arith.CmpiOp(mantissa.result, zero.result, "ne")
        nan = arith.AndIOp(all_exponent.result, nonzero.result)
        quiet_word = arith.OrIOp(word, quiet_bit.result)
        quieted = arith.SelectOp(nan.result, quiet_word.result, word)
        operations.extend([fraction, quiet_bit, word_field, all_exponent, mantissa, nonzero, nan, quiet_word, quieted])
        word = quieted.result
    result = arith.BitcastOp(word, builtin.f32)
    return [*operations, result], result.result


def materialize(op, *, ftz=True):
    """Refine the admitted upstream f32 add event, retaining its exact placement."""
    if (not isinstance(op, memref.AtomicRMWOp) or op.kind.value.data != 0
            or op.results[0].type != builtin.f32 or op.operands[0].type != builtin.f32):
        raise ConversionError("f32 addition requires the original standard typed atomic add")
    outside, value = flush(op.operands[0]) if ftz else ([], op.operands[0])
    body = Block(arg_types=[builtin.f32])
    current_ops, current = flush(body.args[0]) if ftz else ([], body.args[0])
    updated = arith.AddfOp(current, value)
    result_ops, result = flush(updated.result, quiet=True, ftz=ftz)
    body.add_ops([*current_ops, updated, *result_ops, AtomicYieldOp.create(operands=[result])])
    atomic = GenericAtomicRMWOp.create(operands=[*op.operands[1:]], result_types=[builtin.f32], regions=[Region(body)])
    for outer in [*outside, atomic]:
        for nested in outer.walk():
            nested.location = op.location
    Rewriter.replace_op(op, [*outside, atomic], atomic.results)
    return atomic
