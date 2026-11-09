"""Exact integer high products and Triton's Philox word generator."""

import triton
import triton.language as tl


@triton.jit
def high_product_values(x, y, out, n, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    a = tl.load(x + lane, lane < n, other=0)
    b = tl.load(y + lane, lane < n, other=0)
    tl.store(out + lane, tl.umulhi(a, b), lane < n)


@triton.jit
def scalar_high_product(out, a, b, n, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    high = tl.umulhi(a, b)
    tl.store(out + lane, high, lane < n)


@triton.jit
def philox_words(out0, out1, out2, out3, n, seed, counter, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    a, b, c, d = tl.randint4x(seed, counter + lane.to(tl.int64))
    tl.store(out0 + lane, a, lane < n)
    tl.store(out1 + lane, b, lane < n)
    tl.store(out2 + lane, c, lane < n)
    tl.store(out3 + lane, d, lane < n)
