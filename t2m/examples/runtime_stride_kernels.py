"""Independent signed memory progress, source control and final observations."""

import triton
import triton.language as tl


@triton.jit
def invariant_stride(x, out, last, lower, upper, stride, memory_stride,
                     N: tl.constexpr, R: tl.constexpr, B: tl.constexpr,
                     TRANSPOSE: tl.constexpr, MODE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    row_stride = 1 if TRANSPOSE else N
    col_stride = R if TRANSPOSE else 1
    first = x + (16 + lower) * row_stride + lane * col_stride
    second = x + (R - 17 - lower) * row_stride + lane * col_stride
    delta = (stride + memory_stride) * row_stride if MODE == "mixed" else memory_stride * row_stride
    state = tl.load(x + lane * col_stride, lane < N, other=0)
    for row in range(lower, upper, stride):
        if MODE == "pure":
            state = (state ^ (row.to(tl.int64) + 2)) * 3
        else:
            a = tl.load(first, lane < N, other=0)
            b = tl.load(second, lane < N, other=0)
            state += a ^ (b * 3)
        if MODE == "late":
            # upper may be zero at entry. This division executes only inside
            # the original positive-trip body; it must never be speculated.
            delta = (1 // upper) * row_stride
        first += delta
        second -= delta
    final_a = tl.load(first, lane < N, other=0)
    final_b = tl.load(second, lane < N, other=0)
    tl.store(out + lane, state ^ final_a, lane < N)
    tl.store(last + lane, final_a - final_b * 3, lane < N)
