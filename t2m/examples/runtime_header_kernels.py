"""Runtime source loop headers retain row tiles and partial memory state."""

import triton
import triton.language as tl


@triton.jit
def selected_row_sum(x, out, lower, upper, stride, N: tl.constexpr, R: tl.constexpr,
                     B: tl.constexpr, TRANSPOSE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.int64)
    for row in range(lower, upper, stride):
        offset = lane * R + row if TRANSPOSE else row * N + lane
        acc += tl.load(x + offset, lane < N, other=0)
    tl.store(out + lane, acc, lane < N)


@triton.jit
def runtime_partition(x, bias, out, seen, n, lower, upper, stride, B: tl.constexpr,
                      MISSING_ZERO: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    valid = lane < n
    word = tl.load(x + lane, valid, other=0) ^ tl.load(bias)
    positive = word > 0
    tl.store(out + lane, word ^ 1, valid & positive)
    snapshot = word ^ 64
    for iteration in range(lower, upper, stride):
        earlier = tl.load(out + lane, valid & positive, other=-17)
        value = word ^ (iteration.to(tl.int64) + 2) ^ earlier
        tl.store(out + lane, value, valid & ~positive)
        observed = tl.load(out + lane, valid, other=0)
        snapshot ^= observed
        tl.store(out + lane, observed ^ 8, valid & positive)
    if not MISSING_ZERO:  # noqa: SIM102 -- specialize before runtime control.
        if lower >= upper:
            tl.store(out + lane, word ^ 16, valid & ~positive)
    tl.store(seen + lane, snapshot, valid)


@triton.jit
def bidirectional_sum(x, out, last, lower, upper, stride, N: tl.constexpr, R: tl.constexpr,
                      B: tl.constexpr, TRANSPOSE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    row_stride = 1 if TRANSPOSE else N
    col_stride = R if TRANSPOSE else 1
    first = x + lower * row_stride + lane * col_stride
    second = x + (R - 1 - lower) * row_stride + lane * col_stride
    delta = stride * row_stride
    acc = tl.full((B,), 0, tl.int64)
    for row in range(lower, upper, stride):
        a = tl.load(first, lane < N, other=0)
        b = tl.load(second, lane < N, other=0)
        acc += a ^ (b * 3)
        first += delta
        second -= delta
    final_a = tl.load(first, lane < N, other=0)
    final_b = tl.load(second, lane < N, other=0)
    tl.store(out + lane, acc, lane < N)
    tl.store(last + lane, final_a - final_b * 3, lane < N)
