"""Independent rational MX oracle, outside the compiler and its inputs.

OCP MX v1.0 tables 2/5/7 define FP8, FP4 and E8M0. The selected host
profile rounds each f64 product/add and then the declared result. No NumPy
matmul or compiler-generated decode computes the expected contraction.
"""

from fractions import Fraction
import math

import numpy as np

from .float_cast_reference import add_word, convert_word
from .fp8_reference import decode as fp8
from .scalar_math_reference import decode, rounded


def encoded(value, dtype="f64"):
    if math.isnan(value):
        return {"f64": 0x7FF8000000000000, "f32": 0x7FC00000}[dtype]
    if math.isinf(value):
        return {"f64": 0x7FF0000000000000, "f32": 0x7F800000}[dtype] | (
            1 << (63 if dtype == "f64" else 31) if value < 0 else 0
        )
    return rounded(Fraction(value), dtype, negative_zero=bool(np.signbit(value)))


def product(left, right):
    lk, lv = decode(left, "f64")
    rk, rv = decode(right, "f64")
    sign = (left ^ right) & (1 << 63)
    if (
        lk == "nan"
        or rk == "nan"
        or (lk == "inf" and rk == "finite" and not rv)
        or (rk == "inf" and lk == "finite" and not lv)
    ):
        return 0x7FF8000000000000
    if lk == "inf" or rk == "inf":
        return sign | 0x7FF0000000000000
    return rounded(lv * rv, "f64", negative_zero=bool(sign))


def unpack(data, fmt, side, k_pack):
    data = np.asarray(data)
    if fmt == "e2m1":
        axis = data.ndim - (1 if (side == 0) == k_pack else 2)
        shape = list(data.shape)
        shape[axis] *= 2
        output = np.empty(shape, np.float64)
        values = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        for index in np.ndindex(*shape):
            packed = list(index)
            packed[axis] //= 2
            nibble = (int(data[tuple(packed)]) >> (4 * (index[axis] % 2))) & 15
            output[index] = -values[nibble & 7] if nibble & 8 else values[nibble]
        return output
    if fmt in ("e4m3", "e5m2"):
        name = {"e4m3": "f8E4M3FN", "e5m2": "f8E5M2"}[fmt]
        return np.asarray(
            [fp8(int(word), name) for word in data.view(np.uint8).flat]
        ).reshape(data.shape)
    if fmt == "bf16":
        return (
            (data.view(np.uint16).astype(np.uint32) << 16)
            .view(np.float32)
            .astype(np.float64)
        )
    return data.astype(np.float64)


def reference(
    a,
    b,
    sa,
    sb,
    initial,
    lhs="e2m1",
    rhs="e2m1",
    lhs_k=True,
    rhs_k=True,
    scale_formats=("i8", "i8"),
):
    left, right = unpack(a, lhs, 0, lhs_k), unpack(b, rhs, 1, rhs_k)

    def scale_values(values, scales, side, fmt):
        words = np.empty(values.shape, np.uint64)
        for index in np.ndindex(*values.shape):
            value = values[index]
            if scales is None:
                words[index] = encoded(value)
                continue
            group = 32 if fmt == "i8" else 16
            location = (
                *index[:-2],
                index[-2 if side == 0 else -1],
                index[-1 if side == 0 else -2] // group,
            )
            code = int(scales.view(np.uint8)[location])
            factor = (
                (math.nan if code == 255 else Fraction(2) ** (code - 127))
                if fmt == "i8"
                else fp8(code, "f8E4M3FN")
            )
            if not math.isfinite(value) or not math.isfinite(factor):
                with np.errstate(invalid="ignore"):
                    words[index] = encoded(value * float(factor))
            else:
                words[index] = rounded(
                    Fraction(value) * Fraction(factor),
                    "f64",
                    negative_zero=bool(np.signbit(value) != np.signbit(float(factor))),
                )
        return words

    left = scale_values(left, sa, 0, scale_formats[0])
    right = scale_values(right, sb, 1, scale_formats[1])
    words = np.empty(initial.shape, np.uint32)
    for index in np.ndindex(*initial.shape):
        batch, row, col = index[:-2], index[-2], index[-1]
        accumulator = encoded(float(initial[index]))
        for k in range(left.shape[-1]):
            accumulator = add_word(
                accumulator,
                product(int(left[(*batch, row, k)]), int(right[(*batch, k, col)])),
                "f64",
            )
        words[index] = convert_word(accumulator, "f64", "f32", "rtne")
    return words.view(np.float32)
