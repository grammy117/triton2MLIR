"""Real Triton sources for the first exact-tile demo; no GPU execution required."""

import triton
import triton.language as tl


@triton.jit
def vector_add(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)


@triton.jit
def multiply_add(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    product = x * y
    tl.store(out_ptr + offsets, product + x)


@triton.jit
def masked_add(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, mask=offsets < N, other=0.0)
    y = tl.load(y_ptr + offsets, mask=offsets < N, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=offsets < N)


@triton.jit
def masked_multiply_add(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr, N: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, mask=offsets < N, other=0.0)
    y = tl.load(y_ptr + offsets, mask=offsets < N, other=0.0)
    tl.store(out_ptr + offsets, x * y + x, mask=offsets < N)
