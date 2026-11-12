"""Independent source-word evaluator for logical early-return programs."""

import numpy as np


def update(x, state, n, steps, mode):
    assert x.shape == state.shape == (n,) and x.dtype == state.dtype
    width = x.dtype.itemsize * 8
    unsigned = np.dtype(f"uint{width}")
    result = state.copy()
    words = result.view(unsigned)
    data = x.view(unsigned)
    mask = (1 << width) - 1
    for lane in range(n):
        if (lane // 8 + mode) % 2 == 0:
            continue
        word = int(words[lane])
        for i in range(steps):
            word ^= (int(data[lane]) + i + 1) & mask
        words[lane] = word
    return result
