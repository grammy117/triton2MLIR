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
