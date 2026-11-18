"""Preserve an old-value observation while updating tiled caller state."""

import triton
import triton.language as tl


@triton.jit
def observe_and_bump(state, observed, n, bias, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    active = lane < n
    old = tl.load(state + lane, active, other=0)
    tl.store(observed + lane, old, active)
    tl.store(state + lane, old + bias.to(old.dtype), active)
