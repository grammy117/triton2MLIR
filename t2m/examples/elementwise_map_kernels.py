"""Mapped scalar branches, tuple results and loop recurrences."""

import triton
import triton.language as tl


@triton.jit
def _map_scalar(a, b, iterations):
    if b != 0:
        value = a / b if a.dtype.is_floating() else a // b
    else:
        value = tl.full((), -7, a.dtype)
    for step in range(iterations):
        value = value + b * step
    return value, a + b


@triton.jit
def elementwise_map_values(x, y, out_a, out_b, n, iterations, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    a, b = tl.load(x + lane, lane < n, other=0), tl.load(y + lane, lane < n, other=0)
    first, second = tl.map_elementwise(_map_scalar, a, b, iterations)
    tl.store(out_a + lane, first, lane < n)
    tl.store(out_b + lane, second, lane < n)


@triton.jit
def _packed_pair(a0, a1, b0, b1):
    return a0 + b0, a1 - b1


@triton.jit
def packed_map_values(x, y, out):
    lane = tl.arange(0, 8)
    a, b = tl.load(x + lane), tl.load(y + lane)
    value, = tl.map_elementwise(_packed_pair, a, b, pack=2)
    tl.store(out + lane, value)


@triton.jit
def _map_pointer_load(pointer):
    return tl.load(pointer)


@triton.jit
def pointer_load_map(x, out):
    lane = tl.arange(0, 8)
    value = tl.map_elementwise(_map_pointer_load, x + lane)
    tl.store(out + lane, value)
