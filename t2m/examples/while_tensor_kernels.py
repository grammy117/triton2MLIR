"""Original while conditions and ordered memory updates with reusable tiles."""

import triton
import triton.language as tl


@triton.jit
def tiled_while_update(x, state, before, after, n, steps, choose,
                       B: tl.constexpr, MODE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    active = lane < n
    reused = tl.load(x + lane, active, other=0)
    if MODE == "preinit":
        tl.store(state + lane, 0, active)
    if MODE != "fresh":
        tl.store(before + lane, tl.load(state + lane, active, other=0), active)
    else:
        tl.store(before + lane, 0, active)
    i = 0
    while i < steps:
        if MODE != "conditional" or choose:
            if MODE == "fresh":
                value = reused + i.to(reused.dtype)
            else:
                value = tl.load(state + lane, active, other=0) + reused + i.to(reused.dtype)
            tl.store(state + lane, value, active)
        i += 1
    tl.store(after + lane, tl.load(state + lane, active, other=0), active)


@triton.jit
def probe_while(state, checks, condition_seen, limit):
    count = tl.load(checks)
    tl.store(condition_seen, count)
    tl.store(checks, count + 1)
    return tl.load(state) < limit


@triton.jit
def checked_while_update(state, checks, before, after, condition_seen, limit, step):
    tl.store(before, tl.load(state))
    while probe_while(state, checks, condition_seen, limit):
        tl.store(state, tl.load(state) + step)
    tl.store(after, tl.load(state))
