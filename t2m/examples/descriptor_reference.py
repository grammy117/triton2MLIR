"""Independent logical row indexing and allocation-state update oracles.

The compiler never imports this module. No device descriptor, physical address
or expected result participates in conversion or consumer compilation.
"""

import numpy as np


def _values(dtype, shape):
    if dtype == "i64":
        pattern = np.asarray([2**60 + 7, -(2**59) - 3, 0, 17, -29, 2**53 + 1], np.int64)
    elif dtype in ("i16", "i32"):
        width = int(dtype[1:])
        pattern = np.asarray(
            [-(2 ** (width - 1)), 2 ** (width - 1) - 1, -1, 0, 17, -29],
            np.dtype(f"i{width // 8}"),
        )
    elif dtype == "f32":
        pattern = np.asarray(
            [0x7FC01234, 0xFFC05678, 0, 0x80000000, 0x3F800000, 0xBF800000], np.uint32
        ).view(np.float32)
    else:
        words = (
            [0x7FC1, 0xFFD2, 0, 0x8000, 0x3F80, 0xBF80]
            if dtype == "bf16"
            else [0x7E31, 0xFE52, 0, 0x8000, 0x3C00, 0xBC00]
        )
        pattern = np.asarray(words, np.uint16)
        if dtype == "f16":
            pattern = pattern.view(np.float16)
    return np.resize(pattern, shape).copy()


def transfer_arguments(
    kind,
    dtype,
    *,
    rows=7,
    cols=19,
    count=13,
    y=4,
    stride=24,
    padding="zero",
    selected=None,
):
    if selected is None:
        selected = (
            np.resize(
                np.asarray(
                    [-1, 0, rows - 1, rows, 2, 0, -(2**31), 2**31 - 1], np.int32
                ),
                count,
            )
            if kind == "gather"
            else np.arange(count, dtype=np.int32) - 2
        )
    selected = np.asarray(selected, np.int32).copy()
    if selected.shape != (count,):
        raise ValueError("one row index is required per transferred row")
    source = _values(dtype, (rows, stride) if kind == "gather" else (count, 16))
    if kind == "gather":
        expected = np.zeros((count, 16), dtype=source.dtype)
        if padding == "nan":
            expected.fill(0x7FC0 if dtype == "bf16" else np.nan)
        values = [source, selected]
    else:
        initial = _values(dtype, (rows, stride))
        expected = initial.copy()
        values = [source, selected, initial]
    for lane, row in enumerate(selected):
        for column in range(16):
            target_col = y + column
            if 0 <= int(row) < rows and 0 <= target_col < cols:
                if kind == "gather":
                    expected[lane, column] = source[row, target_col]
                else:
                    expected[row, target_col] = source[lane, column]
    return [
        *values,
        *(np.asarray(v, np.int32) for v in (rows, cols, count, y, stride)),
    ], expected


def reduction_arguments(
    kind,
    dtype,
    *,
    rows=3,
    cols=19,
    tiles=3,
    steps=2,
    row=-1,
    y=4,
    stride=24,
    unsigned=False,
):
    if dtype == "f32":
        source = (
            np.arange(tiles * 32, dtype=np.float32).reshape(tiles, 2, 16) % 7 - 3
        ) / 4
        initial = np.arange(rows * stride, dtype=np.float32).reshape(rows, stride) / 2
    else:
        source, initial = _values(dtype, (tiles, 2, 16)), _values(dtype, (rows, stride))
    expected = initial.copy()
    op = {
        "add": np.add,
        "min": np.minimum,
        "max": np.maximum,
        "and": np.bitwise_and,
        "or": np.bitwise_or,
        "xor": np.bitwise_xor,
    }[kind]
    target = (
        expected.view(np.dtype(f"u{expected.dtype.itemsize}")) if unsigned else expected
    )
    operands = source.view(target.dtype)
    for tile in range(tiles):
        for _ in range(steps):
            for i in range(2):
                for j in range(16):
                    if 0 <= row + i < rows and 0 <= y + j < cols:
                        # Ufunc array operands retain the original word width,
                        # including defined modular integer addition.
                        a = target[row + i : row + i + 1, y + j : y + j + 1]
                        a[...] = op(a, operands[tile, i : i + 1, j : j + 1])
    values = [
        source,
        initial,
        *(np.asarray(v, np.int32) for v in (rows, cols, tiles, steps, row, y, stride)),
    ]
    return values, expected


def policy():
    return {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "words"}]},
    }
