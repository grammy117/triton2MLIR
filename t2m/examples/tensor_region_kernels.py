"""Bitset computation with local tensor state inside scalar region bodies."""

import triton
import triton.language as tl


@triton.jit
def _tensor_intersection_for(left, right):
    bit = tl.arange(0, 8)
    state = tl.full((8,), 0, left.dtype)
    for start in range(0, left.dtype.primitive_bitwidth, 8):
        shift = start + bit
        flags = ((left >> shift) & 1) & ((right >> shift) & 1)
        state = state | (flags << shift)
    return tl.sum(state, 0)


@triton.jit
def _tensor_intersection_while(left, right):
    bit = tl.arange(0, 8)
    state = tl.full((8,), 0, left.dtype)
    start = 0
    while start < left.dtype.primitive_bitwidth:
        shift = start + bit
        flags = ((left >> shift) & 1) & ((right >> shift) & 1)
        state = state | (flags << shift)
        start += 8
    return tl.sum(state, 0)


@triton.jit
def _tensor_roundtrip(value):
    bit = tl.arange(0, value.dtype.primitive_bitwidth)
    words = ((value >> bit) & 1) << bit
    prefixes = tl.associative_scan(words, 0, _sum_words)
    last = tl.where(bit == value.dtype.primitive_bitwidth - 1, prefixes, 0)
    return tl.sum(last, 0)


@triton.jit
def _sum_words(left, right):
    return left + right


@triton.jit
def tensor_region_bitsets(x, out, n, B: tl.constexpr, KIND: tl.constexpr,
                         WHILE: tl.constexpr, REVERSE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=-1)
    if KIND == "map":
        result = tl.map_elementwise(_tensor_roundtrip, value)
    elif KIND == "reduce":
        if WHILE:
            result = tl.reduce(value, 0, _tensor_intersection_while)
        else:
            result = tl.reduce(value, 0, _tensor_intersection_for)
    else:
        if WHILE:
            result = tl.associative_scan(value, 0, _tensor_intersection_while, reverse=REVERSE)
        else:
            result = tl.associative_scan(value, 0, _tensor_intersection_for, reverse=REVERSE)
    tl.store(out + lane, result, lane < n)
