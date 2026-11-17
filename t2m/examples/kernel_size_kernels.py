"""A real reduction whose result controls later kernel allocations."""

import triton
import triton.language as tl


@triton.jit
def positive_count(x, count, capacity, B: tl.constexpr):
    lane = tl.arange(0, B)
    value = tl.load(x + lane, lane < capacity, other=0)
    total = tl.sum((value > 0).to(tl.int64), 0)
    tl.store(count, total)


@triton.jit
def positive_pack(x, out, capacity, n, B: tl.constexpr):
    lane = tl.arange(0, B)
    value = tl.load(x + lane, lane < capacity, other=0)
    selected = (lane < capacity) & (value > 0)
    rank = tl.cumsum(selected.to(tl.int32), 0) - 1
    tl.store(out + rank, value, selected & (rank < n))
