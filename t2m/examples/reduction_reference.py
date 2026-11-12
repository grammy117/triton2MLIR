"""Independent source-width scalar reference, with explicit original padding."""

import math

import numpy as np


def recurrence(x, y, n, steps, kind, axis):
    out_a, out_b = x.copy(), y.copy()
    mask = (1 << 64) - 1

    def signed(value):
        word = int(value) & mask
        return word - (1 << 64) if word >> 63 else word

    for start in range(0, n, 8):
        active = min(8, n - start)
        fills = (
            (1, 0)
            if kind == "affine"
            else (np.inf, 2147483647)
            if kind == "argmin"
            else (-1, -1)
            if kind == "and"
            else (0, 0)
        )
        first, second = np.full(8, fills[0], x.dtype), np.full(8, fills[1], y.dtype)
        first[:active], second[:active] = (
            x[start : start + active],
            y[start : start + active],
        )
        first, second = first.reshape(2, 4), second.reshape(2, 4)
        for _ in range(steps):
            new_a, new_b = first.copy(), second.copy()
            groups = (
                ([column, column + 4] for column in range(4))
                if axis == 0
                else (list(range(row * 4, row * 4 + 4)) for row in range(2))
            )
            for indices in groups:
                a, b = first.flat[indices[0]], second.flat[indices[0]]
                for index in indices[1:]:
                    c, d = first.flat[index], second.flat[index]
                    if kind == "affine":
                        a, b = signed(int(a) * int(c)), signed(int(b) * int(c) + int(d))
                    elif kind == "and":
                        a, b = int(a) & int(c), int(b) & int(d)
                    elif kind == "gcd":
                        a, b = (
                            signed(math.gcd(int(a) & mask, int(c) & mask)),
                            signed(math.gcd(int(b) & mask, int(d) & mask)),
                        )
                    else:
                        choose = (
                            np.isnan(a) and (not np.isnan(c) or b <= d)
                            if np.isnan(a) or np.isnan(c)
                            else a < c or a == c and b <= d
                        )
                        if not choose:
                            a, b = c, d
                for index in indices:
                    new_a.flat[index], new_b.flat[index] = a, b
            first, second = new_a, new_b
        out_a[start : start + active], out_b[start : start + active] = (
            first.reshape(-1)[:active],
            second.reshape(-1)[:active],
        )
    return out_a, out_b
