"""Real general reductions with coupled tuples and structured scalar control."""

import triton
import triton.language as tl


@triton.jit
def _and(left, right):
    return left & right


@triton.jit
def _gcd(left, right):
    while right != 0:
        remainder = left % right
        left = right
        right = remainder
    return left


@triton.jit
def _compose(a, b, c, d):
    # Applying (a,b) then (c,d): x -> (a*x+b)*c+d, modulo source width.
    return a * c, b * c + d


@triton.jit
def _argmin(value, index, other_value, other_index):
    left_nan = value != value  # noqa: PLR0124 - IEEE NaN classification in source IR.
    right_nan = other_value != other_value  # noqa: PLR0124
    if left_nan | right_nan:
        left = left_nan & ((~right_nan) | (index <= other_index))
    else:
        left = (value < other_value) | ((value == other_value) & (index <= other_index))
    return tl.where(left, value, other_value), tl.where(left, index, other_index)


@triton.jit
def reduce_recurrence(
    x,
    y,
    out_a,
    out_b,
    n,
    steps,
    B: tl.constexpr,
    KIND: tl.constexpr,
    AXIS: tl.constexpr,
):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if KIND == "affine":
        a = tl.load(x + lane, lane < n, other=1)
        b = tl.load(y + lane, lane < n, other=0)
    elif KIND == "argmin":
        a = tl.load(x + lane, lane < n, other=float("inf"))
        b = tl.load(y + lane, lane < n, other=2147483647)
    else:
        a = tl.load(x + lane, lane < n, other=-1 if KIND == "and" else 0)
        b = tl.load(y + lane, lane < n, other=-1 if KIND == "and" else 0)
    a = tl.reshape(a, (2, B // 2))
    b = tl.reshape(b, (2, B // 2))
    for step in range(steps):
        if KIND == "affine":
            first, second = tl.reduce((a, b), AXIS, _compose)
        elif KIND == "argmin":
            first, second = tl.reduce((a, b), AXIS, _argmin)
        elif KIND == "and":
            first = tl.reduce(a, AXIS, _and)
            second = tl.reduce(b, AXIS, _and)
        else:
            first = tl.reduce(a.to(tl.uint64), AXIS, _gcd).to(a.dtype, bitcast=True)
            second = tl.reduce(b.to(tl.uint64), AXIS, _gcd).to(b.dtype, bitcast=True)
        a = tl.broadcast_to(tl.expand_dims(first, AXIS), (2, B // 2))
        b = tl.broadcast_to(tl.expand_dims(second, AXIS), (2, B // 2))
    tl.store(out_a + lane, tl.reshape(a, (B,)), lane < n)
    tl.store(out_b + lane, tl.reshape(b, (B,)), lane < n)
