"""Independent integer byte addresses around a public typed owning tensor."""

import numpy as np

from ..tensor_storage import DTYPES
from ..typed_storage_contract import WIDTHS
from .allocation_word_reference import patterns
from .byte_buffer_reference import arguments


def typed_arguments(dtype, order, **options):
    size = WIDTHS[dtype]
    capacity = options.pop("capacity", 256)
    origin = options.pop("origin", 9)
    stride = options.pop("stride", 24)
    values, expected = arguments(
        32,
        order,
        capacity=capacity * size,
        origin=origin * size,
        stride=stride * size,
        **options,
    )
    # Unobserved suffix words include zeros, infinities and signaling NaNs.
    samples = patterns(size * 8)
    if capacity >= len(samples) + 16:
        for offset, word in enumerate(samples):
            start = (capacity - len(samples) + offset) * size
            raw = np.frombuffer(int(word).to_bytes(size, order), np.int8)
            values[0][start : start + size] = raw
            expected[0][start : start + size] = raw

    def words(raw):
        endian = "<" if order == "little" else ">"
        unsigned = np.frombuffer(raw.tobytes(), dtype=f"{endian}u{size}")
        return unsigned.astype(f"uint{size * 8}").view(DTYPES[dtype]).copy()

    values[0] = words(values[0])
    expected[0] = words(expected[0])
    values[3] = np.asarray(capacity, np.int32)
    values[4] = np.asarray(origin, np.int32)
    values[5] = np.asarray(stride, np.int32)
    return values, expected
