"""Inactive source bodies retain original state without hiding earlier accesses."""

import triton
import triton.language as tl


@triton.jit
def empty_recurrence(x, bias, out, seen, n, steps, B: tl.constexpr, MODE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    valid = lane < n
    word = tl.load(x + lane, valid, other=0) ^ tl.load(bias)
    positive = word > 0
    if MODE == "early":
        outside = tl.load(out + lane, valid, other=0)
    elif MODE == "bounds":
        outside = tl.load(x + n + lane, valid, other=0)
    else:
        outside = tl.full((B,), 0, tl.int64)
    if MODE == "unsafe-before":
        displacement = lane // steps
    else:
        displacement = lane
    tl.store(out + lane, word ^ 1, valid if MODE == "complete" else valid & positive)
    if MODE == "before":  # noqa: SIM102 -- specialize before actual source control.
        if steps <= 0:
            tl.store(out + lane, word ^ 16, valid & ~positive)
    snapshot = word ^ 64
    for iteration in range(steps):
        if MODE == "address":
            observed = tl.load(x + lane // steps, valid, other=0)
        elif MODE == "unsafe-before":
            observed = tl.load(x + displacement, valid, other=0)
        else:
            # This read may precede complete output initialization; the entire
            # body must be proved inactive rather than granted initialization.
            observed = tl.load(out + lane, valid, other=0)
        snapshot ^= observed ^ outside ^ iteration.to(tl.int64)
        tl.store(out + lane, observed ^ 8, valid)
    if MODE != "before" and MODE != "complete":  # noqa: SIM102
        if steps <= 0:
            tl.store(out + lane, word ^ 16, valid & ~positive)
    tl.store(seen + lane, snapshot, valid)
