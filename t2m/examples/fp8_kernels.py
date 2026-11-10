"""Typed FP8 transport, with original masks, control and source words."""

import triton
import triton.language as tl


@triton.jit
def fp8_choose(x, y, out, n, rounds, B: tl.constexpr, SCALAR: tl.constexpr):
    lane = tl.program_id(0) if SCALAR else tl.program_id(0) * B + tl.arange(0, B)
    zero = tl.full((), 0, tl.uint8).to(x.dtype.element_ty, bitcast=True)
    value = tl.load(x + lane, lane < n, other=zero)
    other = tl.load(y + lane, lane < n, other=zero)
    for step in range(rounds):
        value = tl.where((lane + step) % 3 == 1, other, value)
    tl.store(out + lane, value, lane < n)


@triton.jit
def fp8_widen(x, y, out, raw, n, rounds, B: tl.constexpr, TARGET: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    zero = tl.full((), 0, tl.uint8).to(x.dtype.element_ty, bitcast=True)
    value = tl.load(x + lane, lane < n, other=zero)
    other = tl.load(y + lane, lane < n, other=zero)
    for step in range(rounds):
        value = tl.where((lane + step) % 3 == 1, other, value)
    tl.store(raw + lane, value, lane < n)
    if TARGET == "f16":
        result = value.to(tl.float16)
    elif TARGET == "bf16":
        result = value.to(tl.bfloat16)
    elif TARGET == "f64":
        result = value.to(tl.float64)
    else:
        result = value.to(tl.float32)
    tl.store(out + lane, result, lane < n)
