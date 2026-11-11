"""Independent byte-address oracle for typed pointer/descriptor parameters."""

import numpy as np

from ..tensor_storage import DTYPES
from .allocation_word_reference import patterns


def arguments(
    dtype,
    order,
    *,
    descriptor=False,
    width=32,
    n=7,
    m=2,
    steps=2,
    capacity=128,
    read=9,
    write=10,
    read_row=24,
    read_col=1,
    write_row=24,
    write_col=1,
    invalid=False,
):
    root_size = np.dtype(DTYPES[dtype]).itemsize
    size = width // 8
    memory = (
        (np.arange(capacity * root_size, dtype=np.uint32) * 73 + 211) & 255
    ).astype(np.uint8)
    if descriptor:
        if width != 32:
            raise ValueError("the descriptor source has original f32/i32 words")
        tiles = [
            [
                (i * read_row + j * read_col, i * write_row + j * write_col)
                for i in range(row, min(m, row + 2))
                for j in range(col, min(n, col + 16))
            ]
            for row in range(0, m, 2)
            for col in range(0, n, 16)
        ]
    else:
        tiles = [
            [(i, i) for i in range(first, min(n, first + 8))]
            for first in range(0, n, 8)
        ]

    def address(origin, offset):
        return origin * root_size + offset * size

    def valid(index):
        return 0 <= index and index + size <= len(memory)

    # Original source words include noncanonical NaNs; no float computation.
    samples = patterns(width)
    for tile in tiles:
        for offset, _ in tile:
            index = address(read, offset)
            if valid(index):
                memory[index : index + size] = np.frombuffer(
                    int(samples[offset % len(samples)]).to_bytes(size, order), np.uint8
                )
    expected = memory.copy()
    for tile in tiles:
        for _ in range(steps):
            for toggle in (1, 2):
                snapshots = []
                for first, second in tile:
                    source, target = address(read, first), address(write, second)
                    if not valid(source) or not valid(target):
                        if invalid:
                            continue
                        raise ValueError(
                            "independent source parameter view escapes its typed root"
                        )
                    word = (
                        int.from_bytes(
                            expected[source : source + size].tobytes(), order
                        )
                        ^ toggle
                    )
                    snapshots.append((target, word))
                for target, word in snapshots:
                    expected[target : target + size] = np.frombuffer(
                        word.to_bytes(size, order), np.uint8
                    )

    def typed(raw):
        byte_order = "<" if order == "little" else ">"
        words = np.frombuffer(raw.tobytes(), dtype=f"{byte_order}u{root_size}")
        return words.astype(f"uint{root_size * 8}").view(DTYPES[dtype]).copy()

    if descriptor:
        values = [
            typed(memory),
            *(np.asarray(v, np.int32) for v in (m, n, steps, capacity)),
            *(
                np.asarray(v, np.int64)
                for v in (read, write, read_row, read_col, write_row, write_col)
            ),
        ]
    else:
        values = [
            typed(memory),
            *(np.asarray(v, np.int32) for v in (n, steps, capacity)),
            np.asarray(read, np.int64),
            np.asarray(write, np.int64),
        ]
    return values, typed(expected)
