"""Independent whole-tile snapshot and shared-allocation update oracle."""

import numpy as np

from .descriptor_reference import _values


def arguments(
    dtype,
    *,
    m=2,
    n=13,
    steps=2,
    capacity=160,
    read_origin=8,
    write_origin=16,
    read_strides=(24, 1),
    write_strides=(24, 1),
    invalid=False,
):
    root = _values(dtype, (capacity,))

    def address(origin, row, col, strides):
        index = origin + row * strides[0] + col * strides[1]
        if not 0 <= index < capacity:
            if invalid:
                return None
            raise ValueError("independent descriptor view escapes its allocation")
        return index

    if dtype != "i64":
        for row in range(m):
            for col in range(n):
                index = address(read_origin, row, col, read_strides)
                if index is not None:
                    value = np.asarray(((index * 7) % 31 - 15) / 8, np.float32)
                    root[index] = (
                        int(value.view(np.uint32)) >> 16 if dtype == "bf16" else value
                    )
    expected = root.copy()
    for row_start in range(0, m, 2):
        for col_start in range(0, n, 16):
            for _ in range(steps):
                # Snapshot every active source lane before the first write.
                tile = []
                for i in range(2):
                    for j in range(16):
                        row, col = row_start + i, col_start + j
                        if row >= m or col >= n:
                            continue
                        read = address(read_origin, row, col, read_strides)
                        write = address(write_origin, row, col, write_strides)
                        if read is None or write is None:
                            continue
                        value = expected[read]
                        if dtype == "bf16":
                            value = np.asarray(np.uint32(value) << np.uint32(16)).view(
                                np.float32
                            )
                            summed = np.add(value, np.float32(1), dtype=np.float32)
                            word = int(summed.view(np.uint32))
                            quotient, remainder = divmod(word, 65536)
                            value = quotient + (
                                remainder > 32768 or remainder == 32768 and quotient % 2
                            )
                        elif dtype == "i64":
                            value = np.add(
                                np.asarray(value, np.int64),
                                np.asarray(1, np.int64),
                                dtype=np.int64,
                            )
                        else:
                            value = np.add(value, np.float32(1), dtype=np.float32)
                        tile.append((write, value))
                for write, value in tile:
                    expected[write] = value
    values = [
        root,
        *(np.asarray(value, np.int32) for value in (m, n, steps, capacity)),
        *(
            np.asarray(value, np.int64)
            for value in (read_origin, write_origin, *read_strides, *write_strides)
        ),
    ]
    return values, expected
