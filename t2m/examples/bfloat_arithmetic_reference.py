"""Exact binary-rational BF16 arithmetic, independent of native f32 emulation."""


def _decode(word):
    sign = word >> 15
    exponent, fraction = (word >> 7) & 255, word & 127
    if exponent == 255:
        return ("nan" if fraction else "inf"), 0, sign
    # Every finite BF16 value is an integer multiple of 2**-133.
    integer = ((128 + fraction) << (exponent - 1)) if exponent else fraction
    return "finite", -integer if sign else integer, sign


def _round(integer, scale, sign):
    integer = abs(integer)
    if not integer:
        return sign << 15
    exponent = integer.bit_length() - 1 + scale
    quantum = max(exponent - 7, -133)
    shift = quantum - scale
    if shift > 0:
        quotient, remainder = divmod(integer, 1 << shift)
        halfway = 1 << (shift - 1)
        quotient += remainder > halfway or remainder == halfway and quotient % 2
    else:
        quotient = integer << -shift
    if exponent < -126:
        return (sign << 15) | quotient
    if quotient == 256:
        quotient = 128
        exponent += 1
    if exponent > 127:
        return (sign << 15) | 0x7F80
    return (sign << 15) | ((exponent + 127) << 7) | (quotient - 128)


def arithmetic(left, right, kind):
    """Return exact result bits, or None for the unspecified NaN payload class."""
    if kind == "sub":
        right ^= 0x8000
    lk, lv, ls = _decode(left)
    rk, rv, rs = _decode(right)
    if "nan" in (lk, rk):
        return None
    if kind in ("add", "sub"):
        if lk == rk == "inf" and ls != rs:
            return None
        if lk == "inf":
            return (ls << 15) | 0x7F80
        if rk == "inf":
            return (rs << 15) | 0x7F80
        result = lv + rv
        sign = int(result < 0) if result else (ls if lv == rv == 0 and ls == rs else 0)
        return _round(result, -133, sign)
    if kind != "mul":
        raise ValueError("BF16 oracle supports add, sub and mul")
    sign = ls ^ rs
    if (lk == "inf" and rk == "finite" and rv == 0) or (
        rk == "inf" and lk == "finite" and lv == 0
    ):
        return None
    if "inf" in (lk, rk):
        return (sign << 15) | 0x7F80
    return _round(lv * rv, -266, sign)
