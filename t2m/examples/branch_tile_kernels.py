"""Partial tile prefixes complete through actual scalar branch successors."""

import triton
import triton.language as tl


@triton.jit
def branch_partition(x, bias, out, seen, n, control, B: tl.constexpr, MODE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if MODE == "repeat-program":
        lane = tl.arange(0, B)
    valid = lane < n
    reused = tl.load(bias)
    word = tl.load(x + lane, valid, other=0) ^ reused
    p = word > 0
    q = control > 0 if MODE == "guarded" else control != 0
    tl.store(out + lane, word ^ 1, valid & (~q if MODE == "one-arm" else p))
    if MODE == "prefix-observed" or MODE == "two-roots":
        earlier = tl.load(out + lane, valid & p, other=-17)
    elif MODE == "early-prefix":
        earlier = tl.load(out + lane, valid, other=-17)
    elif MODE == "early-branch":
        earlier = tl.full((B,), -99, tl.int64)
    if MODE == "two-roots":
        tl.store(seen + lane, earlier ^ word ^ 16, valid & p)
    if q:
        if MODE == "branch-observed":
            earlier = tl.load(out + lane, valid & p, other=-17)
        elif MODE == "early-branch":
            earlier = tl.load(out + lane, valid, other=-17)
        if MODE == "nested":
            if control > 0:
                tl.store(out + lane, word ^ 2, valid & ~p)
            else:
                tl.store(out + lane, word ^ 8, valid & ~p)
        elif MODE == "guarded":
            quotient = (word & 65535) // control
            tl.store(out + lane, quotient, valid & ~p)
        elif MODE == "redundant":
            tl.store(out + lane, word ^ 2, valid & ~p & q)
        elif MODE == "one-arm":
            tl.store(out + lane, word ^ 2, valid & q)
        elif MODE == "wrong-index":
            tl.store(out + lane + 1, word ^ 2, valid & ~p & (lane + 1 < n))
        elif MODE == "different-root":
            tl.store(seen + lane, word ^ 2, valid & ~p)
        else:
            tl.store(out + lane, word ^ 2, valid & ~p)
        if MODE == "two-roots":
            tl.store(seen + lane, word ^ 32, valid & ~p)
    elif MODE != "missing-arm" and MODE != "one-arm":
        if MODE == "branch-observed":
            earlier = tl.load(out + lane, valid & p, other=-31)
        if MODE == "redundant":
            tl.store(out + lane, word ^ 4, valid & ~p & ~q)
        else:
            tl.store(out + lane, word ^ 4, valid & ~p)
        if MODE == "two-roots":
            tl.store(seen + lane, word ^ 64, valid & ~p)
    if MODE != "two-roots":
        if MODE != "prefix-observed" and MODE != "branch-observed" and MODE != "early-prefix" and MODE != "early-branch":
            earlier = tl.load(out + lane, valid, other=-17)
        tl.store(seen + lane, earlier, valid)
