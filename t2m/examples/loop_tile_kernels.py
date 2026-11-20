"""Partial tile memory becomes original-loop-carried tensor state."""

import triton
import triton.language as tl


@triton.jit
def loop_partition(x, bias, out, seen, n, steps, B: tl.constexpr, MODE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if MODE == "repeat-program":
        lane = tl.arange(0, B)
    valid = lane < n
    word = tl.load(x + lane, valid, other=0) ^ tl.load(bias)
    p = word > 0
    tl.store(out + lane, word ^ 1, valid if MODE == 'complete' else valid & p)
    if MODE == 'zero-before':  # noqa: SIM102 -- specialize before runtime control.
        if steps <= 0:
            tl.store(out + lane, word ^ 16, valid & ~p)
    snapshot = tl.full((B,), 0, tl.int64)
    for i in range(steps):
        earlier = tl.load(out + lane, valid if MODE == "early-body" else valid & p, other=-17)
        value = (word ^ (i.to(tl.int64) + 2)) ^ earlier
        if MODE == "nested":  # noqa: SIM102 -- retain the source scalar branch.
            if i % 2 != 0:
                value = (word ^ (i.to(tl.int64) + 32)) ^ earlier
        destination = lane + 1 if MODE == "wrong-index" else lane
        mask = valid & ~p & (lane % 2 == 0) if MODE == "incomplete-body" else valid & ~p
        if MODE == "wrong-index":
            mask &= lane + 1 < n
        tl.store(out + destination, value, mask)
        observed = tl.load(out + lane, valid, other=0)
        snapshot ^= observed
        tl.store(out + lane, observed ^ 8, valid & p)
    if MODE == 'zero-after':  # noqa: SIM102 -- specialize before runtime control.
        if steps <= 0:
            tl.store(out + lane, word ^ 16, valid & ~p)
    tl.store(seen + lane, snapshot, valid)
