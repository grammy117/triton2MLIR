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
