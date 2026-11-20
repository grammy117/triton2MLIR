"""Data-dependent tile writes retain tails, observations and reuse."""

import triton
import triton.language as tl


@triton.jit
def tile_partition(x, bias, out, seen, n, control, B: tl.constexpr, MODE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if MODE == "repeat-program":
        lane = tl.arange(0, B)
    elif MODE == "duplicate-lanes":
        lane = tl.program_id(0) * B + tl.arange(0, B) // 2
    valid = lane < n
    reused = tl.load(bias)
    value = tl.load(x + lane, valid, other=0)
    p = value > 0
    q = control != 0
    if MODE == "float":
        first, second = tl.full((B,), 11, tl.int64), tl.full((B,), -23, tl.int64)
    else:
        value = value ^ reused
        first, second = value ^ 1, value ^ 2
    if MODE == "nested":
        tl.store(out + lane, first, valid & p & q)
        tl.store(out + lane, second, valid & p & ~q)
        tl.store(out + lane, value ^ 4, valid & ~p)
    elif MODE == "overlap":
        tl.store(out + lane, first, valid & (p | q))
        tl.store(out + lane, second, valid & (~p | ~q))
    else:
        tl.store(out + lane, first, valid & p)
        if MODE == "observed":
            earlier = tl.load(out + lane, valid & p, other=-17)
        elif MODE == "early":
            earlier = tl.load(out + lane, valid, other=-17)
        if MODE == "different-index":
            tl.store(out + lane + 1, second, valid & ~p & (lane + 1 < n))
        elif MODE == "independent":
            tl.store(out + lane, second, valid & ~q)
        elif MODE != "hole":
            tl.store(out + lane, second, valid & ~p)
    if MODE != "observed" and MODE != "early":
        earlier = tl.load(out + lane, valid, other=-17)
    tl.store(seen + lane, earlier, valid)
