"""BF16 storage transport and explicit source conversions."""

import triton
import triton.language as tl


@triton.jit
def bfloat_copy(x, out, n, B: tl.constexpr, SCALAR: tl.constexpr):
    lane = tl.program_id(0) if SCALAR else tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=0)
    tl.store(out + lane, value, lane < n)


@triton.jit
def bfloat_convert(x, out, n, B: tl.constexpr, NARROW: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=0)
    if NARROW:
        value = value.to(tl.bfloat16, fp_downcast_rounding="rtne")
    else:
        value = value.to(tl.float32)
    tl.store(out + lane, value, lane < n)


@triton.jit
def bfloat_choose(x, y, out, n, rounds, B: tl.constexpr, SCALAR: tl.constexpr):
    lane = tl.program_id(0) if SCALAR else tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=0)
    other = tl.load(y + lane, lane < n, other=0)
    for step in range(rounds):
        value = tl.where((lane + step) % 3 == 1, other, value)
    tl.store(out + lane, value, lane < n)


@triton.jit
def bfloat_observe(x, weight, out, words, value, N: tl.constexpr, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    a = tl.load(x + lane, lane < N, other=0)
    b = tl.load(weight + lane, lane < N, other=0)
    selected = tl.where(lane % 2 == 0, a, b)
    tl.store(out + lane, selected, lane < N)
    tl.store(words + lane, selected.to(tl.int16, bitcast=True), lane < N)
    tl.store(value + lane, a.to(tl.float32) + b.to(tl.float32), lane < N)
