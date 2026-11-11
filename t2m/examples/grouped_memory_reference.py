"""Independent value oracle for the example's declared initial source profile.

This evaluator imports no compiler, layout witness or emitted operation.
Logical program boundaries constrain the changing-state observations.
"""

import numpy as np


def reference(x, y, n, capacity, iterations, delta, *, warps, selected):
    x, y = np.asarray(x), np.asarray(y)
    if (x.dtype != y.dtype or x.shape != y.shape or x.size != capacity
            or not 0 <= n <= capacity or not 0 <= iterations <= 3 or warps not in (2, 4)):
        raise ValueError("unqualified example memory domain")
    flat, other = x.reshape(-1), y.reshape(-1)
    stride, tile = warps * 32, 256
    total = ((n + tile - 1) // tile) * tile
    choice = np.arange(capacity) % 3 == 0 if selected else np.zeros(capacity, dtype=bool)

    def read(state):
        values = np.full(total, -7, dtype=x.dtype)
        values[:n] = np.where(choice[:n], other[:n], state[:n])
        pairs = values.reshape(-1, 2, stride)
        a, b = pairs[:, 0] + pairs[:, 1], pairs[:, 1] - pairs[:, 0]
        if n:
            right = np.arange(total).reshape(-1, 2, stride)[:, 1]
            roots = right % 3 == 0 if selected else np.zeros_like(right, dtype=bool)
            for step in range(iterations):
                index = right + step
                active = (right < n) & (index < capacity) & (index // tile == right // tile)
                safe = np.minimum(index, capacity - 1)
                value = np.where(roots, other[safe], state[safe])
                a = a + np.where(active, value, x.dtype.type(-3))
        return values[:n].copy(), np.stack((a, b), axis=1).reshape(-1)[:n].copy()

    initial, first = read(flat)
    changed = flat.copy()
    changed[:n] = initial + x.dtype.type(delta)
    final, second = read(changed)
    return changed.reshape(x.shape), initial, first, final, second
