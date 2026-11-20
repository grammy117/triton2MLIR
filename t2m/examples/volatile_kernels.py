"""Repeated private volatile observations retain every original load event."""

import triton
import triton.language as tl


@triton.jit
def volatile_updates(x, state, old, seen, n, steps, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    active = lane < n
    value = tl.load(x + lane, active, other=0, volatile=True)
    _unused = tl.load(x + lane, active, other=0, volatile=True)
    initial = tl.full((B,), -7, state.dtype.element_ty)
    tl.store(old + lane, initial, active)
    tl.store(seen + lane, initial, active)
    for _ in range(steps):
        before = tl.load(state + lane, active, other=-3, volatile=True)
        tl.store(state + lane, before + value, active)
        after = tl.load(state + lane, active, other=-3, volatile=True)
        tl.store(old + lane, before, active)
        tl.store(seen + lane, after, active)
