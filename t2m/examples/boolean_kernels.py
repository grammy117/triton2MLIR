"""Logical Boolean memory with source masks and inactive destination state."""

import triton
import triton.language as tl


@triton.jit
def boolean_memory(x, out, n, B: tl.constexpr, FILL: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, (lane < n) & (lane % 3 != 1), other=FILL)
    tl.store(out + lane, value, (lane < n) & (lane % 4 != 2))
