"""Associative scan combiners retain actual scalar branches and loop state."""

import triton
import triton.language as tl


@triton.jit
def _segmented_sum(left, left_head, right, right_head):
    if right_head:
        value = right
    else:
        value = left + right
    return value, left_head | right_head


@triton.jit
def segmented_scan(
    x, heads, out, combined_heads, n, B: tl.constexpr, REVERSE: tl.constexpr
):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=0)
    head = tl.load(heads + lane, lane < n, other=0)
    scanned, flags = tl.associative_scan(
        (value, head), 0, _segmented_sum, reverse=REVERSE
    )
    tl.store(out + lane, scanned, lane < n)
    tl.store(combined_heads + lane, flags, lane < n)


@triton.jit
def _gcd(left, right):
    while right != 0:
        remainder = left % right
        left = right
        right = remainder
    return left


@triton.jit
def _loop_product(left, right):
    result = tl.full((), 0, left.dtype)
    for bit in range(left.dtype.primitive_bitwidth):
        if (right & 1) != 0:
            result = result + left
        left = left << 1
        right = right >> 1
    return result


@triton.jit
def loop_scan(x, out, n, B: tl.constexpr, REVERSE: tl.constexpr, GCD: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=0 if GCD else 1)
    if GCD:
        result = tl.associative_scan(value, 0, _gcd, reverse=REVERSE)
    else:
        result = tl.associative_scan(value, 0, _loop_product, reverse=REVERSE)
    tl.store(out + lane, result, lane < n)
