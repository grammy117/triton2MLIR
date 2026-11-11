"""Independent word/rounding oracle; never imported by compiler lowering."""

import numpy as np


def reference(x, n, capacity, iterations, delta):
    original = np.asarray(x)
    flat = original.reshape(-1)
    if capacity != flat.size or not 0 <= n <= capacity or iterations < 0:
        raise ValueError("mapped read reference requires its declared valid source domain")

    def add(left, right):
        if original.dtype.kind in "iu":
            bits = original.dtype.itemsize * 8
            word = (int(left) + int(right)) & ((1 << bits) - 1)
            return np.asarray(word, dtype=f"uint{bits}").view(original.dtype)[()]
        return original.dtype.type(left + right)

    def read(state):
        result = state[:n].copy()
        for step in range(iterations):
            for lane in range(n):
                value = state[lane + step] if lane + step < capacity else original.dtype.type(-3)
                result[lane] = add(result[lane], value)
        return result

    changed = flat.copy()
    for lane in range(n):
        changed[lane] = add(changed[lane], delta)
    return [changed.reshape(original.shape), flat[:n].copy(), read(flat), changed[:n].copy(), read(changed)]
