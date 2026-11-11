"""Packed computation whose logical meaning needs the original register groups."""

import triton
import triton.language as tl


@triton.jit
def _mix2(a0, a1, b0, b1):
    return a0 + a1, a1 - a0, a0 + b0, a1 - b1


@triton.jit
def _mix4(a0, a1, a2, a3, b0, b1, b2, b3):
    return a0 + a3, a1 - a2, a2 + a1, a3 - a0, a0 + b0, a1 - b1, a2 + b2, a3 - b3


@triton.jit
def grouped_map(x, y, out_mix, out_asymmetric, n, B: tl.constexpr, PACK: tl.constexpr, MATRIX: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if MATRIX:
        lane = tl.reshape(lane, (16, B // 16))
    a = tl.load(x + lane, lane < n, other=0)
    b = tl.load(y + lane, lane < n, other=0)
    if PACK == 2:
        first, second = tl.map_elementwise(_mix2, a, b, pack=PACK)
    else:
        first, second = tl.map_elementwise(_mix4, a, b, pack=PACK)
    tl.store(out_mix + lane, first, lane < n)
    tl.store(out_asymmetric + lane, second, lane < n)


@triton.jit
def _control2(a0, a1, b0, b1):
    if b0 != 0:
        value = a0 // b0
    else:
        value = a1
    step = 0
    while step < (b1 & 3):
        value += a1
        step += 1
    return value, value + b1, a0 + b0, a1 - b1


@triton.jit
def grouped_control_map(x, y, out_mix, out_asymmetric, n, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    a = tl.load(x + lane, lane < n, other=0)
    b = tl.load(y + lane, lane < n, other=0)
    first, second = tl.map_elementwise(_control2, a, b, pack=2)
    tl.store(out_mix + lane, first, lane < n)
    tl.store(out_asymmetric + lane, second, lane < n)
