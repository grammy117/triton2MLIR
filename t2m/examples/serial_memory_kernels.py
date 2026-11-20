"""Serial indexed state and source-proved complete scalar result loops."""

import triton
import triton.language as tl


@triton.jit
def single_ranked_tile(x, out):
    row = tl.arange(0, 1)[:, None]
    column = tl.arange(0, 1)[None, :]
    value = tl.load(x + row + column)
    tl.store(out + row + column, value + 3)


@triton.jit
def serial_snapshot(state, before, after, seen, n, count, position, enabled, MODE: tl.constexpr):
    if MODE == "before-read":
        value = tl.load(before)
        tl.store(seen, value)
    for i in range(count if MODE == "different" else n):
        if MODE == "masked":
            tl.store(before + i, tl.load(state + i), i < n - 1)
        elif MODE == "branch":
            if i % 2 == 0:
                tl.store(before + i, tl.load(state + i))
        elif MODE == "duplicate":
            tl.store(before + i % 2, tl.load(state + i))
        else:
            tl.store(before + i, tl.load(state + i))
    old = tl.load(state + position, enabled != 0, other=-19)
    tl.store(state + position + (n if MODE == "bad-store" else 0), old + 7, enabled != 0)
    new = tl.load(state + position, enabled != 0, other=-23)
    tl.store(seen, old)
    tl.store(seen + 1, new)
    for i in range(n):
        tl.store(after + i, tl.load(before + i))
