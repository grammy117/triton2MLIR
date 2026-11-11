"""Logical descriptor indexing oracle; never imported by conversion code."""

import itertools
import math

import numpy as np

from .descriptor_reference import _values


def _round_tf32(value):
    word = int(np.asarray(value, np.float32).view(np.uint32))
    if word & 0x7F800000 == 0x7F800000:
        return value
    magnitude = word & 0x7FFFFFFF
    quotient, remainder = divmod(magnitude, 8192)
    if remainder > 4096 or (remainder == 4096 and quotient % 2):
        quotient += 1
    return np.asarray((word & 0x80000000) | (quotient * 8192), np.uint32).view(
        np.float32
    )[()]


def arguments(
    kind,
    dtype,
    *,
    block=(2, 16),
    dimensions=(7, 19),
    strides=(24, 1),
    count=13,
    tiles=3,
    steps=2,
    offsets=(0, 4, 0),
    capacity=256,
    padding_nan=False,
    round_tf32=False,
    selected=None,
    invalid=False,
):
    transfer = kind in ("gather", "scatter")
    block_op = kind in ("load", "store")
    block = (1, 16) if transfer else tuple(block) if block_op else (2, 16)
    rank = len(block)
    if len(dimensions) != rank or len(strides) != rank:
        raise ValueError(
            "descriptor oracle requires one logical dimension/stride per axis"
        )
    root = _values(dtype, (capacity,))
    if dtype == "f32":
        # Includes TF32 ties, subnormals, overflow, signs and exact special words.
        words = np.asarray(
            [
                0x3F801000,
                0x3F803000,
                0xBF801000,
                0x007FFFFF,
                0x00001000,
                0x7F7FF000,
                0x7F800000,
                0xFFC01234,
                0x7F801234,
                0x80000000,
            ],
            np.uint32,
        )
        root = np.resize(words, capacity).view(np.float32)
    writable = kind not in ("gather", "load")
    if not transfer and not block_op and dtype == "f32":
        root = np.arange(capacity, dtype=np.float32) / 2
    value_shape = (
        (count, 16) if transfer else (math.prod(block),) if block_op else (tiles, 2, 16)
    )
    source = _values(dtype, value_shape)
    if not transfer and not block_op and dtype == "f32":
        source = (
            np.arange(tiles * 32, dtype=np.float32).reshape(value_shape) % 7 - 3
        ) / 4
    expected = root.copy() if writable else np.zeros(value_shape, root.dtype)
    if not writable and padding_nan and not dtype.startswith("i"):
        expected.fill(0x7FC0 if dtype == "bf16" else np.nan)
    if transfer:
        if selected is None:
            selected = (
                np.resize(
                    np.asarray(
                        [
                            -1,
                            0,
                            dimensions[0] - 1,
                            dimensions[0],
                            2,
                            0,
                            -(2**31),
                            2**31 - 1,
                        ],
                        np.int32,
                    ),
                    count,
                )
                if kind == "gather"
                else np.arange(count, dtype=np.int32) - 2
            )
        selected = np.asarray(selected, np.int32).copy()
        if selected.shape != (count,):
            raise ValueError("one descriptor row index is required per row")

    def address(coordinates):
        if not all(
            0 <= coord < size
            for coord, size in zip(coordinates, dimensions, strict=True)
        ):
            return None
        result = sum(
            coord * stride for coord, stride in zip(coordinates, strides, strict=True)
        )
        if not 0 <= result < capacity:
            if invalid:
                return None  # Only the independent abort oracle is used in these cases.
            raise ValueError("active oracle address outside allocation")
        return result

    def loaded(index):
        value = root[index]
        return _round_tf32(value) if dtype == "f32" and round_tf32 else value

    if transfer:
        for lane, row in enumerate(selected):
            for col in range(16):
                target = address((int(row), offsets[1] + col))
                if target is not None:
                    if writable:
                        expected[target] = source[lane, col]
                    else:
                        expected[lane, col] = loaded(target)
    elif block_op:
        for flat, coordinate in enumerate(
            itertools.product(*(range(n) for n in block))
        ):
            target = address(tuple(offsets[i] + n for i, n in enumerate(coordinate)))
            if target is not None:
                if writable:
                    expected[target] = source[flat]
                else:
                    expected[flat] = loaded(target)
    else:
        operation = {
            "add": np.add,
            "min": np.minimum,
            "max": np.maximum,
            "and": np.bitwise_and,
            "or": np.bitwise_or,
            "xor": np.bitwise_xor,
        }[kind]
        for tile in range(tiles):
            for _ in range(steps):
                for i in range(2):
                    for j in range(16):
                        target = address((offsets[0] + i, offsets[1] + j))
                        if target is not None:
                            expected[target : target + 1] = operation(
                                expected[target : target + 1],
                                source[tile, i, j : j + 1],
                            )
    scalars = [
        *(np.asarray(value, np.int64) for value in (*dimensions, *strides)),
        np.asarray(padding_nan),
        np.asarray(round_tf32),
        *(np.asarray(value, np.int32) for value in dimensions),
        *(np.asarray(value, np.int64) for value in strides),
    ]
    ordinary = (
        (count, offsets[1], capacity)
        if transfer
        else (*offsets, capacity)
        if block_op
        else (tiles, steps, offsets[0], offsets[1], capacity)
    )
    scalars += [np.asarray(value, np.int32) for value in ordinary]
    return [
        root,
        *([source] if writable else []),
        *([selected] if transfer else []),
        *scalars,
    ], expected


def matmul_arguments(m, n, k, *, dtype="bf16", strides=None, sensitive=False):
    strides = strides or (k + 3, 1, n + 5, 1)
    ar, ak, bk, bc = strides

    def allocation(rows, cols, row_stride, col_stride):
        if not rows or not cols:
            return np.empty(0, np.uint16 if dtype == "bf16" else np.float32), np.empty(
                (rows, cols), np.float32
            )
        count = (rows - 1) * row_stride + (cols - 1) * col_stride + 8
        values = (
            np.ones(count, np.float32)
            if sensitive
            else (np.arange(count, dtype=np.float32) % 17 - 8) / 8
        )
        root = (
            (values.view(np.uint32) >> 16).astype(np.uint16)
            if dtype == "bf16"
            else values
        )
        view = np.ndarray(
            (rows, cols),
            dtype=root.dtype,
            buffer=root,
            strides=(row_stride * root.itemsize, col_stride * root.itemsize),
        )
        logical = (
            (view.astype(np.uint32) << 16).view(np.float32)
            if dtype == "bf16"
            else view.copy()
        )
        return root, logical

    a, left = allocation(m, k, ar, ak)
    b, right = allocation(k, n, bk, bc)
    initial = (
        np.full((m, n), 2.0**24, np.float32)
        if sensitive
        else np.resize(
            np.asarray([2.0**24, -(2.0**24), 1.0, -0.0, 0.0, 3.0], np.float32), (m, n)
        )
    )
    output = initial.copy()
    for inner in range(((k + 31) // 32) * 32):
        product = (
            left[:, inner : inner + 1] * right[inner : inner + 1, :]
            if inner < k
            else np.zeros_like(output)
        )
        output = np.add(output, product, dtype=np.float32)
    fields = []
    for dims, steps in (((m, k), (ar, ak)), ((k, n), (bk, bc))):
        fields.extend(
            [
                *(np.asarray(value, np.int64) for value in (*dims, *steps)),
                np.asarray(False),
                np.asarray(False),
                *(np.asarray(value, np.int32) for value in dims),
                *(np.asarray(value, np.int64) for value in steps),
            ]
        )
    return [
        a,
        b,
        initial,
        *fields,
        *(np.asarray(value, np.int32) for value in (m, n, k, a.size, b.size)),
    ], output
