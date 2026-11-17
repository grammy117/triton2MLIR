"""A real reduction whose result controls later kernel allocations."""

import triton
import triton.language as tl


@triton.jit
def positive_count(x, count, capacity, B: tl.constexpr):
    lane = tl.arange(0, B)
    value = tl.load(x + lane, lane < capacity, other=0)
    total = tl.sum((value > 0).to(tl.int64), 0)
    tl.store(count, total)
