"""Independent unsigned-word snapshots for differently typed allocation views."""

import numpy as np


def patterns(width):
    fraction = {16: 10, 32: 23, 64: 52}[width]
    sign = 1 << (width - 1)
    infinity = ((1 << (width - 1 - fraction)) - 1) << fraction
    words = [
        0,
        sign,
        infinity,
        infinity | sign,
        infinity | (1 << (fraction - 1)) | 53,
        infinity | sign | (1 << (fraction - 1)) | 86,
        infinity | 17,
        infinity | sign | 34,
        1,
        sign | 1,
        infinity - 1,
        (infinity - 1) | sign,
    ]
    if width == 16:
        # Include BF16's own NaN, infinity and largest finite encodings too.
        words += [0x7F80, 0xFF80, 0x7FC1, 0xFFC2, 0x7F81, 0xFF82, 0x7F7F, 0xFF7F]
    return np.asarray(words, getattr(np, f"uint{width}"))


def arguments(
    width=32,
    storage="integer",
    bfloat=False,
    *,
    n=7,
    steps=2,
    capacity=96,
    read=8,
    write=10,
    invalid=False,
):
    words = np.resize(patterns(width), capacity)
    expected = words.copy()
    for start in range(0, n, 8):
        lanes = np.arange(start, min(n, start + 8))
        reads, writes = read + lanes, write + lanes
        if invalid and (
            np.any(reads < 0)
            or np.any(reads >= capacity)
            or np.any(writes < 0)
            or np.any(writes >= capacity)
        ):
            continue
        for _ in range(steps):
            snapshot = expected[reads].copy()
            expected[writes] = snapshot ^ 1
            snapshot = expected[reads].copy()
            expected[writes] = snapshot ^ 2
    dtype = getattr(
        np,
        f"int{width}"
        if storage == "integer"
        else f"uint{width}"
        if bfloat
        else f"float{width}",
    )
    return [
        words.view(dtype),
        *(np.asarray(v, np.int32) for v in (n, steps, capacity)),
        np.asarray(read, np.int64),
        np.asarray(write, np.int64),
    ], expected.view(dtype)


def descriptor_arguments(*, m=2, n=13, steps=2, capacity=160, read=8, write=16):
    words = np.resize(patterns(32), capacity)
    expected = words.copy()
    for row in range(0, m, 2):
        for col in range(0, n, 16):
            indices = np.asarray(
                [
                    i * 24 + j
                    for i in range(row, min(m, row + 2))
                    for j in range(col, min(n, col + 16))
                ],
                np.int64,
            )
            for _ in range(steps):
                snapshot = expected[read + indices].copy()
                expected[write + indices] = snapshot ^ 1
                snapshot = expected[read + indices].copy()
                expected[write + indices] = snapshot ^ 2
    return [
        words.view(np.int32),
        *(np.asarray(v, np.int32) for v in (m, n, steps, capacity)),
        *(np.asarray(v, np.int64) for v in (read, write, 24, 1, 24, 1)),
    ], expected.view(np.int32)
