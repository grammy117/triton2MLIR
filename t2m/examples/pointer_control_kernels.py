"""Logical pointers crossing structured regions, without physical addresses."""

import triton
import triton.language as tl


@triton.jit
def while_pointer_sum(x, out, n, rows, steps, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    pointer = x + lane
    acc = tl.full((B,), 0, tl.float32)
    step = 0
    while step < steps:
        acc += tl.load(pointer, lane < n, other=0)
        pointer += n
        step += 1
    acc += tl.load(pointer, lane < n, other=0)
    tl.store(out + lane, acc, lane < n)


@triton.jit
def branch_pointer_select(x, out, n, choice, B: tl.constexpr, SCALAR: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    if choice > 0:
        pointer = x + lane
    else:
        pointer = x + n + lane
    value = tl.load(pointer, lane < n, other=0)
    tl.store(out + lane, value, lane < n)


@triton.jit
def mixed_base_branch(x, y, out, n, choice, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if choice > 0:
        pointer = x + lane
    else:
        pointer = y + lane
    value = tl.load(pointer, lane < n, other=0)
    tl.store(out + lane, value, lane < n)


@triton.jit
def branch_pointer_words(x, encoded, out, n, choice, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    key = tl.load(encoded + lane, lane < n, other=0)
    pointer = x + lane
    if choice > 0:
        pointer = pointer + key
        pointer = pointer + key
        pointer = pointer - 2 * key.to(tl.int64)
    else:
        word = key * 4
        pointer = pointer + word
        pointer = pointer - 4 * lane.to(tl.int64)
    value = tl.load(pointer, lane < n, other=0)
    tl.store(out + lane, value, lane < n)
