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


@triton.jit
def update_matrix_views(left, right, old_left, old_right, bias_ptr,
                        left_n, right_n, left_stride, right_stride,
                        left_k, right_k, left_k_stride, right_k_stride):
    bias = tl.load(bias_ptr)
    for i in range(right_n * right_k):
        offset = (i // right_k) * right_stride + (i % right_k) * right_k_stride
        value = tl.load(right + offset)
        tl.store(old_right + i, value)
    for i in range(left_n * left_k):
        offset = (i // left_k) * left_stride + (i % left_k) * left_k_stride
        value = tl.load(left + offset)
        tl.store(old_left + i, value)
        tl.store(left + offset, value + bias)
    for i in range(right_n * right_k):
        offset = (i // right_k) * right_stride + (i % right_k) * right_k_stride
        value = tl.load(right + offset)
        tl.store(right + offset, value * 3)
