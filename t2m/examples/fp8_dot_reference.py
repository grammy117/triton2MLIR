"""Independent source-ordered FP8 dot reference with exact rational products."""

from fractions import Fraction
import math

import numpy as np

from .float_cast_reference import add_word
from .fp8_reference import decode
from .scalar_math_reference import rounded


def product(left, right):
    negative = bool(np.signbit(left) != np.signbit(right))
    sign = (1 << 31) if negative else 0
    if (
        math.isnan(left)
        or math.isnan(right)
        or (math.isinf(left) and right == 0)
        or (math.isinf(right) and left == 0)
    ):
        return 0x7FC00000
    if math.isinf(left) or math.isinf(right):
        return 0x7F800000 | sign
    return rounded(Fraction(left) * Fraction(right), "f32", negative_zero=negative)


def reference(a, b, initial, lhs, rhs, *, block=32):
    m, k = a.shape
    assert b.shape[0] == k
    n = b.shape[1]
    assert initial.shape == (m, n)
    left = [[decode(int(word), lhs) for word in row] for row in a]
    right = [[decode(int(word), rhs) for word in row] for row in b]
    words = initial.view(np.uint32).copy()
    # Source masks supply typed zero for the padded K lanes in every tile.
    # Those additions remain observable for signed zero and infinity/NaN.
    for row in range(m):
        for col in range(n):
            value = int(words[row, col])
            for inner in range(((k + block - 1) // block) * block):
                term = product(left[row][inner], right[inner][col]) if inner < k else 0
                value = add_word(value, term)
            words[row, col] = value
    return words.view(np.float32)


def inputs(lhs, rhs, m, n, k):
    # Every format encoding appears across the two operands of a full case,
    # including signed zeros and the distinct nonfinite representations.
    a = np.resize(np.arange(256, dtype=np.uint8), (m, k))
    b = np.resize(np.arange(255, -1, -1, dtype=np.uint8), (k, n))
    initial = np.resize(
        np.asarray([0.0, -0.0, 2.0**24, -(2.0**24), 1.0, -3.0], np.float32), (m, n)
    )
    return a, b, initial
