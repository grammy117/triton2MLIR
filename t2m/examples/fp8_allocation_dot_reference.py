"""Independent rational cast and ordered partial-dot allocation reference."""

import numpy as np

from .dot_precision_reference import reference as dot
from .fp8_cast_reference import reference as cast
from .fp8_reference import widened


def reference(left, right, initial, precision):
    def quantize(value, target, mode, policy):
        return widened(cast(value, "f32", target, mode, policy), target, "f32")

    state = initial.copy()
    for start in range(0, left.shape[1], 32):
        a, b = left[:, start : start + 32], right[start : start + 32, :]
        with np.errstate(invalid="ignore", over="ignore"):
            a = a + quantize(a, "f8E5M2", "rtne", "ieee")
            b = b + quantize(b, "f8E4M3FN", "rtz", "satfinite")
            state = state + quantize(state, "f8E5M2", "rtz", "satfinite-nan-inf")
        state = dot(a, b, state, precision)
    return state
