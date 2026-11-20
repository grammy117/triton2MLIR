"""While-carried word indices with original modular numeric state."""

import triton
import triton.language as tl


@triton.jit
def prefix_while(x, state, before, after, n):
    tl.store(before, tl.load(state))
    i = 0
    while i < n:
        value = tl.load(x + i)
        tl.store(state, tl.load(state) ^ (value + i.to(tl.int64)))
        i += 1
    tl.store(after, tl.load(state))


@triton.jit
def wrapped_while(x, state, before, after, start, steps, delta, ADDRESS: tl.constexpr):
    tl.store(before, tl.load(state))
    i, count = start, 0
    while count < steps:
        if ADDRESS == "wrap":
            offset = ((i % 15) + 15) % 15
        elif ADDRESS == "add":
            offset = i + 3
        elif ADDRESS == "divide":
            offset = 16 // i
        elif ADDRESS == "direct":
            offset = i
        else:
            offset = i.to(tl.int32)
        value = tl.load(x + offset)
        tl.store(state, tl.load(state) ^ value)
        i += delta
        count += 1
    final = ((i % 15) + 15) % 15
    tl.store(after, tl.load(state) ^ tl.load(x + final))
