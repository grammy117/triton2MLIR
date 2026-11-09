"""Conditional reduction state with a nonzero initial value."""

import triton
import triton.language as tl


@triton.jit
def conditional_reduction(x, out, n, steps, choice, B: tl.constexpr, R: tl.constexpr):
    row = tl.arange(0, R)
    col = tl.program_id(0) * B + tl.arange(0, B)
    tile = tl.load(x + row[:, None] * n + col[None, :], col[None, :] < n, other=0)
    reduced = tl.sum(tile, axis=0)
    acc = tl.full((B,), 2.5, tl.float32)
    step = 0
    while step < steps:
        if (step + choice) % 2 == 0:
            acc = reduced
        else:
            acc = tl.sum(tile + acc[None, :], axis=0)
        step += 1
    tl.store(out + col, acc, col < n)
