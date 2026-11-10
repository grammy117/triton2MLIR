"""Independent format-value oracle; never imported by compiler code.

Format definitions: LLVM a47bddcc APFloat.cpp and APFloatTest.cpp, plus the
pinned Triton test_conversions.py format table. Rational arithmetic supplies
finite values; computed NaN payloads are deliberately not constrained.
"""

from fractions import Fraction
import math

import numpy as np

FORMATS = {
    "f8E4M3FN": (4, 3, 7, "finite-nan"),
    "f8E4M3FNUZ": (4, 3, 8, "unsigned-zero"),
    "f8E5M2": (5, 2, 15, "ieee"),
    "f8E5M2FNUZ": (5, 2, 16, "unsigned-zero"),
}


def decode(word, dtype):
    exponent_bits, fraction_bits, bias, special = FORMATS[dtype]
    if not 0 <= word < 256:
        raise ValueError("FP8 word must fit in one byte")
    negative = bool(word & 128)
    exponent = (word & 127) >> fraction_bits
    fraction = word & ((1 << fraction_bits) - 1)
    if (
        special == "unsigned-zero"
        and word == 128
        or special == "finite-nan"
        and word & 127 == 127
        or special == "ieee"
        and exponent == (1 << exponent_bits) - 1
        and fraction
    ):
        return math.nan
    if special == "ieee" and exponent == (1 << exponent_bits) - 1:
        return -math.inf if negative else math.inf
    if exponent == fraction == 0:
        return -0.0 if negative else 0.0
    significand = fraction if exponent == 0 else (1 << fraction_bits) + fraction
    power = (1 if exponent == 0 else exponent) - bias - fraction_bits
    magnitude = Fraction(significand) * Fraction(2) ** power
    return float(-magnitude if negative else magnitude)


def widened(words, source, target):
    words = np.asarray(words)
    values = np.asarray(
        [decode(int(word), source) for word in np.asarray(words).flat], np.float64
    ).reshape(words.shape)
    if target == "bf16":
        # Every finite FP8 value is exact in BF16. No arithmetic narrowing
        # reference or compiler implementation supplies the expected words.
        return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return values.astype(
        {"f16": np.float16, "f32": np.float32, "f64": np.float64}[target]
    )
