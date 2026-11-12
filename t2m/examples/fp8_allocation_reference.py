"""Independent declared-profile references for FP8 allocation recurrences."""

import numpy as np

from .fp8_cast_reference import reference as cast_reference
from .fp8_reference import decode


def rounded(values):
    result = np.asarray(values, np.float32).copy()
    with np.errstate(invalid="ignore", over="ignore"):
        for target, mode, policy in (
            ("f8E5M2", "rtne", "ieee"),
            ("f8E4M3FN", "rtz", "satfinite"),
            ("f8E5M2", "rtz", "satfinite-nan-inf"),
        ):
            words = cast_reference(values, "f32", target, mode, policy)
            quantized = np.asarray(
                [decode(int(word), target) for word in words.flat], np.float32
            ).reshape(words.shape)
            result += quantized
    return result


def update(
    state,
    m,
    n,
    steps,
    capacity,
    read_origin,
    write_origin,
    read_row,
    read_col,
    write_row,
    write_col,
):
    if np.asarray(state).size != capacity:
        raise ValueError("source allocation value differs from declared capacity")
    result = np.asarray(state, np.float32).copy()
    flat = result.reshape(-1)
    for row in range(0, m, 2):
        for col in range(0, n, 16):
            active = [
                (i, j)
                for i in range(row, min(m, row + 2))
                for j in range(col, min(n, col + 16))
            ]
            reads = [read_origin + i * read_row + j * read_col for i, j in active]
            writes = [write_origin + i * write_row + j * write_col for i, j in active]
            for _ in range(steps):
                flat[writes] = rounded(flat[reads].copy())
    return result
