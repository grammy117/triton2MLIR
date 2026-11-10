"""Independent rational reference for Triton's BF16 composite dot algorithm."""

import numpy as np

from .float_cast_reference import add_word, convert_word
from .scalar_math_reference import decode, rounded


def split(words, count):
    residual = np.asarray(words, np.uint32).copy()
    output = []
    for part in range(count):
        narrow = np.asarray(
            [convert_word(int(word), "f32", "bf16", "rtne") for word in residual.flat],
            np.uint16,
        ).reshape(residual.shape)
        widened = np.asarray(
            [convert_word(int(word), "bf16", "f32", "rtne") for word in narrow.flat],
            np.uint32,
        ).reshape(residual.shape)
        output.append(widened)
        if part + 1 < count:
            residual = np.asarray(
                [
                    add_word(int(a), int(b) ^ (1 << 31))
                    for a, b in zip(residual.flat, widened.flat, strict=True)
                ],
                np.uint32,
            ).reshape(residual.shape)
    return output


def product(left, right):
    lk, lv = decode(left, "f32")
    rk, rv = decode(right, "f32")
    sign = (left ^ right) & (1 << 31)
    if (
        lk == "nan"
        or rk == "nan"
        or lk == "inf"
        and rk == "finite"
        and not rv
        or rk == "inf"
        and lk == "finite"
        and not lv
    ):
        return 0x7FC00000
    if lk == "inf" or rk == "inf":
        return sign | 0x7F800000
    return rounded(lv * rv, "f32", negative_zero=bool(sign))


def reference(a, b, initial, precision, *, block=32):
    count = 2 if precision == "bf16x3" else 3
    terms = ([(1, 1), (2, 0), (0, 2)] if count == 3 else []) + [(1, 0), (0, 1)]
    m, k = a.shape
    n = b.shape[1]
    output = initial.view(np.uint32).copy()
    for start in range(0, k, block):
        left, right = np.zeros((m, block), np.float32), np.zeros((block, n), np.float32)
        stop = min(k, start + block)
        left[:, : stop - start], right[: stop - start] = a[:, start:stop], b[start:stop]
        lhs, rhs = (
            split(left.view(np.uint32), count),
            split(right.view(np.uint32), count),
        )
        state = np.zeros((m, n), np.uint32)

        def dot(ai, bi):
            for row in range(m):
                for col in range(n):
                    value = int(state[row, col])
                    for inner in range(block):
                        value = add_word(
                            value,
                            product(int(lhs[ai][row, inner]), int(rhs[bi][inner, col])),
                        )
                    state[row, col] = value

        for ai, bi in terms:
            dot(ai, bi)
        for index in np.ndindex(*state.shape):
            if decode(int(state[index]), "f32")[0] == "nan":
                state[index] = 0
        dot(0, 0)
        output = np.asarray(
            [
                add_word(int(value), int(c))
                for value, c in zip(state.flat, output.flat, strict=True)
            ],
            np.uint32,
        ).reshape(m, n)
    return output.view(np.float32)


def inputs(m, n, k):
    words = np.asarray(
        [
            0,
            0x80000000,
            0x3F800000,
            0x3F808001,
            0x3F7FFFFF,
            0xBF808001,
            0x00010000,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0x7F7F0000,
            0x7F800000,
            0xFF800000,
            0x7FC01234,
            0x1F800001,
            0x5F7FFFFF,
        ],
        np.uint32,
    )
    a = np.resize(words, (m, k)).view(np.float32)
    b = np.resize(words[::-1], (k, n)).view(np.float32)
    c = np.resize(
        np.asarray([0.0, -0.0, 2**24, -(2**24), 1.0, -3.0], np.float32), (m, n)
    )
    return a, b, c
