"""Independent rational floating conversion reference, outside the compiler."""

from fractions import Fraction

import numpy as np

from t2m.tensor_storage import DTYPES

from .scalar_math_reference import arrays, decode, fields, log2_floor, pack, rounded


def convert_word(word, source, target, mode):
    if mode not in ("rtne", "rtz"):
        raise ValueError("reference rounding must be RTNE or RTZ")
    kind, value = decode(word, source)
    _, _, _, _, source_sign = fields(source)
    _, fraction, bias, infinite, sign = fields(target)
    negative = bool(int(word) & source_sign)
    if kind == "nan":
        return infinite | 1 << (fraction - 1)
    if kind == "inf":
        return infinite | (sign if negative else 0)
    if mode == "rtne":
        return rounded(value, target, negative_zero=negative)
    if not value:
        return sign if negative else 0
    value = abs(value)
    exponent = max(log2_floor(value), 1 - bias)
    scaled = value / Fraction(2) ** (exponent - fraction)
    significand = scaled.numerator // scaled.denominator
    if exponent > bias:
        return infinite - 1 | (sign if negative else 0)
    return pack(significand, exponent, negative, target)


def reference(values, source, target, mode):
    exponent, fraction, *_ = fields(source)
    words = np.asarray(values).view(f"uint{1 + exponent + fraction}")
    result = [convert_word(int(word), source, target, mode) for word in words.flat]
    return arrays(target, result).reshape(words.shape)


def add_word(left, right, dtype="f32"):
    lk, lv = decode(left, dtype)
    rk, rv = decode(right, dtype)
    _, fraction, _, inf, sign = fields(dtype)
    if (
        lk == "nan"
        or rk == "nan"
        or lk == rk == "inf"
        and bool(left & sign) != bool(right & sign)
    ):
        return inf | 1 << (fraction - 1)
    if lk == "inf":
        return left
    if rk == "inf":
        return right
    return rounded(lv + rv, dtype, negative_zero=bool(left & right & sign))


def recurrence(values, delta, steps, target, mode="rtz"):
    delta_word = int(np.asarray(delta, np.float32).view(np.uint32))
    output = []
    for initial in np.asarray(values).view(np.uint32).flat:
        word = int(initial)
        for _ in range(steps):
            word = add_word(word, delta_word)
            narrowed = convert_word(word, "f32", target, mode)
            word = convert_word(narrowed, target, "f32", "rtne")
        output.append(convert_word(word, "f32", target, mode))
    return arrays(target, output).reshape(values.shape)


def inputs(dtype):
    exponent, fraction, bias, inf, sign = fields(dtype)
    if dtype in ("f16", "bf16"):
        return arrays(dtype, np.arange(65536, dtype=np.uint16))
    words = {
        0,
        sign,
        1,
        sign + 1,
        inf,
        inf | sign,
        inf + 1,
        inf + (1 << (fraction - 1)),
        inf - 1,
        inf + sign + 1,
        inf + sign + (1 << (fraction - 1)),
        inf + sign - 1,
    }
    # Every exponent with several significant/rounding boundary patterns.
    for exp in range(1 << exponent):
        for mantissa in (0, 1, (1 << fraction) - 1, 1 << (fraction - 1)):
            words.update(
                (exp << fraction | mantissa, sign | exp << fraction | mantissa)
            )
    if dtype in ("f32", "f64"):
        # Adjacent source words on both sides of exact rounding midpoints.
        points = [
            1 + 2.0**-8,
            1 + 3 * 2.0**-8,
            1 + 2.0**-11,
            1 + 3 * 2.0**-11,
            2.0**-134,
            2.0**-25,
            65520.0,
            (2 - 2.0**-8) * 2.0**127,
        ]
        if dtype == "f64":
            points.extend(
                [1 + 2.0**-24, 1 + 3 * 2.0**-24, 2.0**-150, (2 - 2.0**-24) * 2.0**127]
            )
        midpoints = np.asarray(points, DTYPES[dtype]).view(
            np.uint32 if dtype == "f32" else np.uint64
        )
        for midpoint in midpoints:
            words.update(int(midpoint) + offset for offset in (-2, -1, 0, 1, 2))
            words.update(
                (int(midpoint) + offset) | sign for offset in (-2, -1, 0, 1, 2)
            )
    return arrays(dtype, sorted(words))
