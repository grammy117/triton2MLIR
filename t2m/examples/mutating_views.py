"""Ordered state updates through distinct, possibly overlapping source views."""

import triton
import triton.language as tl


@triton.jit
def update_views(left, right, old_left, old_right, bias_ptr, left_n, right_n, left_stride, right_stride):
    bias = tl.load(bias_ptr)
    for i in range(right_n):
        value = tl.load(right + i * right_stride)
        tl.store(old_right + i, value)
    for i in range(left_n):
        value = tl.load(left + i * left_stride)
        tl.store(old_left + i, value)
        tl.store(left + i * left_stride, value + bias)
    for i in range(right_n):
        value = tl.load(right + i * right_stride)
        tl.store(right + i * right_stride, value * 3)
