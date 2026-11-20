"""Interleaved fresh tiles retain cross-output observations and input state."""

import triton
import triton.language as tl


@triton.jit
def interleaved_partition(x, bias, a, b, seen_a, seen_b, n, control, B: tl.constexpr, MODE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if MODE == "repeat-program":
        lane = tl.arange(0, B)
    valid = lane < n
    reused = tl.load(bias)
    word = tl.load(x + lane, valid, other=0)
    p = word > 0
    q = p ^ (control == 0)
    value = word ^ reused
    tl.store(a + lane, value ^ 1, valid & p)
    if MODE == "cross" or MODE == "predicate":
        earlier_a = tl.load(a + lane, valid & p, other=-17)
        if MODE == "predicate":
            q = earlier_a > 0
        first_b = earlier_a ^ value ^ 2
    else:
        first_b = value ^ 2
    if MODE == "alias-root":
        tl.store(a + lane, first_b, valid & q)
    else:
        tl.store(b + lane, first_b, valid & q)
    if MODE == "cross" or MODE == "predicate" or MODE == "observed" or MODE == "early":
        earlier_a = tl.load(a + lane, valid & p, other=-17)
        if MODE == "early":
            earlier_b = tl.load(b + lane, valid, other=-31)
        else:
            earlier_b = tl.load(b + lane, valid & q, other=-31)
    if MODE == "readonly" or MODE == "state":
        reloaded = tl.load(x + lane, valid, other=0)
        value = reloaded ^ reused
        if MODE == "state":
            tl.store(x + lane, value ^ 16, valid)
            value = tl.load(x + lane, valid, other=0)
    if MODE == "cross" or MODE == "predicate":
        second_a, second_b = earlier_b ^ value ^ 4, earlier_a ^ value ^ 8
    else:
        second_a, second_b = value ^ 4, value ^ 8
    if MODE == "different-index":
        tl.store(a + lane + 1, second_a, valid & ~p & (lane + 1 < n))
    else:
        tl.store(a + lane, second_a, valid & ~p)
    if MODE != "hole":
        tl.store(b + lane, second_b, valid & ~q)
    if MODE != "cross" and MODE != "predicate" and MODE != "observed" and MODE != "early":
        earlier_a = tl.load(a + lane, valid, other=-17)
        earlier_b = tl.load(b + lane, valid, other=-31)
    tl.store(seen_a + lane, earlier_a, valid)
    tl.store(seen_b + lane, earlier_b, valid)
