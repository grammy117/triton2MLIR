"""Source-width interval facts for division, without changing executable IR.

The endpoint/sign partition follows MLIR's InferIntRangeCommon.cpp at
a47bddccec30255619bb8c37fa59700e661d4e66. Here an unproved poison domain
refuses address admission instead of supplying an unconstrained machine range.
"""

from .diagnostics import ConversionError


def division_bounds(left, right, *, width, opcode="arith.divsi"):
    minimum, maximum = -(1 << (width - 1)), (1 << (width - 1)) - 1
    if not all(minimum <= low <= high <= maximum for low, high in (left, right)):
        raise ConversionError(
            "source division operands exceed their signed width", opcode=opcode
        )
    if right[0] <= 0 <= right[1]:
        raise ConversionError("source division can reach a zero divisor", opcode=opcode)
    if left[0] == minimum and right[0] <= -1 <= right[1]:
        raise ConversionError("source signed division can overflow", opcode=opcode)

    def quotient(a, b):
        if opcode == "arith.floordivsi":
            return a // b
        if opcode == "arith.ceildivsi":
            return -((-a) // b)
        if opcode != "arith.divsi":
            raise ValueError(
                "signed division proof requires divsi, floordivsi or ceildivsi"
            )
        result = abs(a) // abs(b)
        return -result if (a < 0) != (b < 0) else result

    corners = [quotient(a, b) for a in left for b in right]
    return min(corners), max(corners)


def remainder_bounds(left, right, *, width, opcode="arith.remsi"):
    """Bound signed remainder using LLVM's sign/modulus partition.

    Keep the native signed-division overflow gate as well as the zero-divisor
    gate: a portable address proof cannot grant an LLVM srem poison input.
    """
    try:
        division_bounds(left, right, width=width, opcode="arith.divsi")
    except ConversionError as error:
        raise ConversionError(str(error), opcode=opcode) from error
    modulus = max(abs(value) for value in right)
    low = max(left[0], 1 - modulus) if left[0] < 0 else 0
    high = min(left[1], modulus - 1) if left[1] > 0 else 0
    if right[0] == right[1] and left[1] - left[0] < modulus:
        def rem(value):
            result = abs(value) % modulus
            return -result if value < 0 else result
        first, last = rem(left[0]), rem(left[1])
        if first <= last:
            low, high = first, last
    return low, high
