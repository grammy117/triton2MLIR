"""Typed source literal transport, independent of target storage placement."""

import triton
import triton.language as tl


@triton.jit
def literal_copy(literal, out, N: tl.constexpr, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(literal + lane, lane < N, other=0)
    tl.store(out + lane, value, lane < N)
