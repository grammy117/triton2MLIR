"""Source allocation recurrences with explicit caller-selected FP8 contracts."""

import triton
import triton.language as tl


@triton.jit
def mixed_round(value):
    first = value.to(tl.float8e5, fp_downcast_rounding="rtne").to(tl.float32)
    second = value.to(tl.float8e4nv, fp_downcast_rounding="rtz").to(tl.float32)
    third = value.to(tl.float8e5, fp_downcast_rounding="rtz").to(tl.float32)
    return ((value + first) + second) + third


@triton.jit
def descriptor_fp8_updates(first, second, m, n, steps, capacity):
    row = tl.program_id(0) * 2
    col = tl.program_id(1) * 16
    for _ in range(steps):
        value = first.load([row, col])
        second.store([row, col], mixed_round(value))


@triton.jit
def pointer_fp8_updates(first, second, n, steps, capacity):
    lane = tl.program_id(0) * 16 + tl.arange(0, 16)
    for _ in range(steps):
        value = tl.load(first + lane, lane < n, other=0)
        tl.store(second + lane, mixed_round(value), lane < n)
