"""Coarse numeric computation remains executable inside scalar map bodies."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _histogram_word(value):
    bit = tl.arange(0, value.dtype.primitive_bitwidth)
    flags = (value >> bit) & 1
    counts = tl.histogram(flags, 2)
    return tl.sum(counts * tl.arange(0, 2), 0).to(value.dtype)


@triton.jit
def _dot_word(left, right, initial):
    a = tl.full((16, 16), left, tl.float32)
    b = tl.full((16, 16), right, tl.float32)
    c = tl.full((16, 16), initial, tl.float32)
    product = tl.dot(a, b, c, input_precision="ieee")
    return tl.sum(tl.sum(product, 0), 0)


@triton.jit
def _dot_zero_word(left, right, ignored):
    return _dot_word(left, right, 0.0)


@triton.jit
def _scaled_word(left, right, initial):
    a = tl.full((16, 32), left, tl.float32).to(tl.float16)
    b = tl.full((32, 16), right, tl.float32).to(tl.bfloat16)
    c = tl.full((16, 16), initial, tl.float32)
    product = tl.dot_scaled(a, None, "fp16", b, None, "bf16", acc=c)
    return tl.sum(tl.sum(product, 0), 0)


@triton.jit
def _gather_word(value):
    index = tl.arange(0, 8)
    values = value + index.to(value.dtype)
    shift = (value & 7).to(tl.int32)
    positions = (index + shift) & 7
    selected = tl.gather(values, positions, 0)
    return tl.sum(selected * (index + 1).to(value.dtype), 0)


@triton.jit
def _logical_word(value):
    return value + tl.program_id(0) * 17 + tl.num_programs(0)


@triton.jit
def _logical_control_word(value):
    result = value
    if value & 1:
        result = result + tl.program_id(0) * 17
    for _ in range((value & 3).to(tl.int32)):
        result = result + tl.num_programs(0) + tl.program_id(1) + tl.num_programs(2)
    return result


@triton.jit
def _math_word(value):
    values = tl.full((8,), value, value.dtype)
    return tl.sum(libdevice.tanh(values), 0)


@triton.jit
def numeric_region_values(x, y, initial, out, n, B: tl.constexpr, FAMILY: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    a = tl.load(x + lane, lane < n, other=0)
    if FAMILY == "histogram":
        result = tl.map_elementwise(_histogram_word, a)
    elif FAMILY == "gather":
        result = tl.map_elementwise(_gather_word, a)
    elif FAMILY == "logical":
        result = tl.map_elementwise(_logical_word, a)
    elif FAMILY == "logical-control":
        result = tl.map_elementwise(_logical_control_word, a)
    elif FAMILY == "math":
        result = tl.map_elementwise(_math_word, a)
    else:
        b = tl.load(y + lane, lane < n, other=0)
        c = tl.load(initial + lane, lane < n, other=0)
        if FAMILY == "dot":
            result = tl.map_elementwise(_dot_word, a, b, c)
        elif FAMILY == "dot-zero":
            result = tl.map_elementwise(_dot_zero_word, a, b, c)
        else:
            result = tl.map_elementwise(_scaled_word, a, b, c)
    tl.store(out + lane, result, lane < n)
