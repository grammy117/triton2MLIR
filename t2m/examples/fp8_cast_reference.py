"""Independent FP8 quantization oracle; never imported by compiler code.

Search the exact rational target palette rather than reproducing the compiler's
significand algorithm. The extra overflow neighbor establishes the final RN
midpoint, including E4M3FN's even maximum. Computed NaN payloads are unspecified.
"""

from bisect import bisect_left, bisect_right
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np

from .fp8_reference import FORMATS, decode as decode_fp8
from .scalar_math_reference import decode, fields

MAX_WORD = {"f8E4M3FN": 126, "f8E4M3FNUZ": 127, "f8E5M2": 123, "f8E5M2FNUZ": 127}
NAN_WORD = {"f8E4M3FN": 127, "f8E4M3FNUZ": 128, "f8E5M2": 126, "f8E5M2FNUZ": 128}


@lru_cache(None)
def palette(dtype):
    values = tuple(
        Fraction(decode_fp8(word, dtype)) for word in range(MAX_WORD[dtype] + 1)
    )
    # The next ideal finite encoding, before the format's special code.
    next_value = values[-1] + (values[-1] - values[-2])
    return values, (values[-1] + next_value) / 2


def convert_word(word, source, target, mode, overflow):
    if mode not in ("rtne", "rtz") or overflow not in (
        "ieee",
        "satfinite",
        "satfinite-nan-inf",
    ):
        raise ValueError("reference requires explicit rounding and overflow policies")
    if source in FORMATS:
        decoded = decode_fp8(int(word), source)
        negative = bool(int(word) & 128)
        kind = (
            "nan" if math.isnan(decoded) else "inf" if math.isinf(decoded) else "finite"
        )
        value = Fraction(decoded) if kind == "finite" else None
    else:
        kind, value = decode(word, source)
        negative = bool(int(word) & fields(source)[-1])
    sign = 128 if negative else 0
    if kind == "nan":
        return NAN_WORD[target]
    if kind == "inf":
        if overflow == "satfinite":
            return MAX_WORD[target] | sign
        if overflow == "ieee" and target == "f8E5M2":
            return 124 | sign
        return NAN_WORD[target]
    values, threshold = palette(target)
    magnitude = abs(value)
    if magnitude > values[-1]:
        if overflow != "ieee" or mode == "rtz":
            result = MAX_WORD[target]
        elif (
            magnitude < threshold
            or magnitude == threshold
            and MAX_WORD[target] % 2 == 0
        ):
            result = MAX_WORD[target]
        else:
            return (124 | sign) if target == "f8E5M2" else NAN_WORD[target]
    elif mode == "rtz":
        result = max(0, bisect_right(values, magnitude) - 1)
    else:
        upper = bisect_left(values, magnitude)
        if upper == 0 or values[upper] == magnitude:
            result = upper
        else:
            lower = upper - 1
            distances = magnitude - values[lower], values[upper] - magnitude
            result = (
                lower
                if distances[0] < distances[1]
                or distances[0] == distances[1]
                and lower % 2 == 0
                else upper
            )
    if result == 0 and target.endswith("FNUZ"):
        return 0
    return result | sign


def reference(values, source, target, mode="rtne", overflow="ieee"):
    values = np.asarray(values)
    width = 8 if source in FORMATS else 1 + sum(fields(source)[:2])
    words = values.view(f"uint{width}")
    return np.asarray(
        [
            convert_word(int(word), source, target, mode, overflow)
            for word in words.flat
        ],
        np.uint8,
    ).reshape(values.shape)


def inputs(source, target):
    from .float_cast_reference import inputs as standard_inputs

    if source in FORMATS:
        return np.arange(256, dtype=np.uint8)
    values = standard_inputs(source)
    if source in ("f16", "bf16"):
        return values
    finite, threshold = palette(target)
    midpoints = [float((left + right) / 2) for left, right in zip(finite, finite[1:])]
    points = np.asarray(
        [*map(float, finite), *midpoints, float(threshold)], values.dtype
    )
    width = int(source[1:])
    sign = 1 << (width - 1)
    words = set(map(int, values.view(f"uint{width}")))
    for point in points.view(f"uint{width}"):
        for offset in (-1, 0, 1):
            word = int(point) + offset
            if word >= 0:
                words.update((word, word | sign))
    return np.asarray(sorted(words), dtype=f"uint{width}").view(values.dtype)


def recurrence(values, delta, steps, target, mode, overflow):
    from .float_cast_reference import add_word

    delta_word = int(np.asarray(delta, np.float32).view(np.uint32))
    output = []
    for initial in np.asarray(values, np.float32).view(np.uint32).flat:
        word = int(initial)
        for _ in range(steps):
            word = add_word(word, delta_word)
            quantized = convert_word(word, "f32", target, mode, overflow)
            value = np.asarray(decode_fp8(quantized, target), np.float32)
            word = int(value.view(np.uint32))
        output.append(convert_word(word, "f32", target, mode, overflow))
    return np.asarray(output, np.uint8).reshape(values.shape)
