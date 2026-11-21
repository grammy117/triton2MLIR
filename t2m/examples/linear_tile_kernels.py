"""Multidimensional computation tiles write an actual flat allocation."""

import triton
import triton.language as tl


@triton.jit
def linear_tile_update(x, out, n, B: tl.constexpr, PARTIAL: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    lane = tl.reshape(lane, (2, B // 2))
    value = tl.load(x + lane, lane < n, other=0)
    value = value * 2 + 1
    mask = lane < n
    if PARTIAL:
        mask = mask & (value > 0)
    tl.store(out + lane, value, mask)
