"""Independent Python words and allocation snapshots, outside compiler inputs."""

import numpy as np


def combine(values, shift):
    values = np.asarray(values)
    width = values.dtype.itemsize * 8
    mask, modulus = (1 << width) - 1, 1 << width
    count = min(int(shift) & ((1 << 32) - 1), width)

    def signed(value):
        value &= mask
        return value - modulus if value >= modulus // 2 else value

    output = []
    for source in values.flat:
        word = int(source)
        right = signed(word ^ 7)
        first = signed(((word & mask) << count) ^ (right & mask))
        second = signed((right >> count) + (word < right))
        output.append(signed(first - 3 * second))
    return np.asarray(output, values.dtype).reshape(values.shape)


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
    *,
    shift=0,
    words=False,
):
    result = np.asarray(state).copy()
    if result.size != capacity or result.dtype not in (
        np.dtype("int32"),
        np.dtype("int64"),
    ):
        raise ValueError("source word allocation must match its complete signed root")
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
                snapshot = flat[reads].copy()
                flat[writes] = (
                    combine(snapshot, shift)
                    if words
                    else snapshot + np.asarray(1, snapshot.dtype)
                )
    return result


def arguments(
    dtype="i64",
    *,
    words=False,
    m=2,
    n=13,
    steps=2,
    capacity=256,
    read_origin=8,
    write_origin=16,
    read_row=24,
    read_col=1,
    write_row=24,
    write_col=1,
    shift=3,
):
    width = 64 if dtype == "i64" else 32
    pattern = [
        0,
        1,
        -1,
        (1 << (width - 1)) - 1,
        -(1 << (width - 1)),
        (1 << (width - 1)) - 2,
        -(1 << (width - 1)) + 1,
        (1 << 53) + 1 if width == 64 else (1 << 24) + 1,
    ]
    root = np.resize(
        np.asarray(pattern, dtype="int64" if width == 64 else "int32"), capacity
    )
    fields = (
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
    )
    expected = update(root, *fields, shift=shift, words=words)
    values = [
        root,
        *(np.asarray(value, np.int32) for value in fields[:4]),
        *(np.asarray(value, np.int64) for value in fields[4:]),
    ]
    if words:
        values.append(np.asarray(shift, np.int32))
    return values, expected
