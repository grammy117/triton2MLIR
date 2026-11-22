"""Exact rational FP8 arithmetic oracle; never imported by compiler code.

Reuse the independently decoded palette. Rounding searches neighboring values;
sqrt compares exact squared midpoints rather than calling floating sqrt.
"""

import math
from bisect import bisect_left
from fractions import Fraction
from functools import lru_cache

from .fp8_cast_reference import MAX_WORD, NAN_WORD, palette
from .fp8_reference import decode


@lru_cache(None)
def value(word, dtype):
    decoded = decode(int(word), dtype)
    return decoded if not math.isfinite(decoded) else Fraction(decoded)


def nan(dtype):
    return NAN_WORD[dtype]


def infinity(negative, dtype):
    return (124 | (128 if negative else 0)) if dtype == "f8E5M2" else nan(dtype)


def zero(negative, dtype):
    return 128 if negative and not dtype.endswith("FNUZ") else 0


def rounded(number, dtype, *, negative_zero=False):
    if not number:
        return zero(negative_zero, dtype)
    negative = number < 0
    magnitude = abs(number)
    values, threshold = palette(dtype)
    if magnitude > values[-1]:
        if magnitude > threshold or magnitude == threshold and MAX_WORD[dtype] % 2:
            return infinity(negative, dtype)
        result = MAX_WORD[dtype]
    else:
        upper = bisect_left(values, magnitude)
        if not upper or values[upper] == magnitude:
            result = upper
        else:
            lower = upper - 1
            midpoint = (values[lower] + values[upper]) / 2
            result = lower if magnitude < midpoint or magnitude == midpoint and lower % 2 == 0 else upper
    return zero(negative, dtype) if not result else result | (128 if negative else 0)


def binary(left, right, kind, dtype):
    a, b = value(int(left), dtype), value(int(right), dtype)
    if math.isnan(a) or math.isnan(b):
        return nan(dtype)
    an, bn = bool(int(left) & 128), bool(int(right) & 128)
    ai, bi = math.isinf(a), math.isinf(b)
    if kind in ("add", "sub"):
        if kind == "sub":
            b, bn = -b, not bn
        if ai and bi and an != bn:
            return nan(dtype)
        if ai or bi:
            return infinity(an if ai else bn, dtype)
        return rounded(a + b, dtype, negative_zero=an and bn)
    negative = an != bn
    if kind == "mul":
        if ai and not b or bi and not a:
            return nan(dtype)
        if ai or bi:
            return infinity(negative, dtype)
        return rounded(a * b, dtype, negative_zero=negative)
    if kind == "div":
        if ai and bi or not a and not b:
            return nan(dtype)
        if ai or not b:
            return infinity(negative, dtype)
        if bi or not a:
            return zero(negative, dtype)
        return rounded(a / b, dtype, negative_zero=negative)
    raise ValueError("unknown reference arithmetic")


@lru_cache(maxsize=1024)
def square_root(word, dtype):
    number = value(int(word), dtype)
    if math.isnan(number) or number < 0:
        return nan(dtype)
    if math.isinf(number):
        return infinity(False, dtype)
    if not number:
        return zero(bool(int(word) & 128), dtype)
    values, _ = palette(dtype)
    upper = bisect_left([v * v for v in values], number)
    assert upper < len(values)
    if not upper or values[upper] ** 2 == number:
        return upper
    lower = upper - 1
    midpoint_squared = ((values[lower] + values[upper]) / 2) ** 2
    return lower if number < midpoint_squared or number == midpoint_squared and lower % 2 == 0 else upper


def fused(left, right, addend, dtype):
    a, b, c = (value(int(word), dtype) for word in (left, right, addend))
    if any(math.isnan(v) for v in (a, b, c)) or math.isinf(a) and not b or math.isinf(b) and not a:
        return nan(dtype)
    negative = bool((int(left) ^ int(right)) & 128)
    if math.isinf(a) or math.isinf(b):
        if math.isinf(c) and negative != bool(int(addend) & 128):
            return nan(dtype)
        return infinity(negative, dtype)
    if math.isinf(c):
        return infinity(bool(int(addend) & 128), dtype)
    return rounded(a * b + c, dtype, negative_zero=negative and bool(int(addend) & 128))


def clamp(word, lower, upper, dtype, *, propagate):
    number = value(int(word), dtype)
    if math.isnan(number):
        return nan(dtype) if propagate else int(lower)
    def key(bits):
        decoded = value(int(bits), dtype)
        assert not math.isnan(decoded)
        return decoded, -1 if not decoded and int(bits) & 128 else 0
    return min(max(int(word), int(lower), key=key), int(upper), key=key)
