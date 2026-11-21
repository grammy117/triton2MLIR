"""Selected per-iteration memory steps, local arithmetic and observations."""

import triton
import triton.language as tl


@triton.jit
def conditional_stride(x, out, last, lower, upper, stride, memory_stride,
                       N: tl.constexpr, R: tl.constexpr, B: tl.constexpr,
                       TRANSPOSE: tl.constexpr, MODE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    row_stride = 1 if TRANSPOSE else N
    col_stride = R if TRANSPOSE else 1
    first = x + (16 + lower) * row_stride + lane * col_stride
    second = x + (R - 17 - lower) * row_stride + lane * col_stride
    state = tl.load(x + lane * col_stride, lane < N, other=0)
    for row in range(lower, upper, stride):
        if MODE == "pure":
            state = (state ^ (row.to(tl.int64) + 2)) * 3
        else:
            state += tl.load(first, lane < N, other=0) ^ (tl.load(second, lane < N, other=0) * 3)
        if MODE == "bit":
            delta = (row < 3).to(tl.int64) * row_stride
            first += delta
            second -= delta
        elif MODE == "scalar":
            if row < 3:
                delta = memory_stride * row_stride
            else:
                delta = (1 // (row - 2)) * row_stride
            first += delta
            second -= delta
        else:
            if row < 3:
                first += memory_stride * row_stride
                second -= memory_stride * row_stride
            else:
                if MODE == "nested":
                    if row < 5:
                        first += (1 // (row - 2)) * row_stride
                        second -= (1 // (row - 2)) * row_stride
                    else:
                        first -= memory_stride * row_stride
                        second += memory_stride * row_stride
                else:
                    first += (1 // (row - 2)) * row_stride
                    second -= (1 // (row - 2)) * row_stride
        if MODE == "observe":
            state ^= tl.load(first, lane < N, other=0)
    final_a = tl.load(first, lane < N, other=0)
    final_b = tl.load(second, lane < N, other=0)
    tl.store(out + lane, state ^ final_a, lane < N)
    tl.store(last + lane, final_a - final_b * 3, lane < N)


@triton.jit
def conflicting_choice(x, out, B: tl.constexpr):
    """Invalid parallel ownership: programs one and two overlap a result tile."""
    lane = tl.arange(0, B)
    program = tl.program_id(0)
    if program < 2:
        offset = program * B
    else:
        offset = B
    tl.store(out + offset + lane, tl.load(x + lane))


@triton.jit
def conflicting_loop_choice(x, out, upper, B: tl.constexpr):
    """A loop-invariant step still differs between logical programs."""
    lane = tl.arange(0, B)
    program = tl.program_id(0)
    advance = tl.minimum(program, 1)
    position = 0
    for iteration in range(upper):
        position += advance
    tl.store(out + (program + 1 - position) * B + lane, tl.load(x + lane))


@triton.jit
def conflicting_branch_loop_choice(x, out, upper, B: tl.constexpr):
    lane = tl.arange(0, B)
    program = tl.program_id(0)
    if program == 0:
        advance = 1
    else:
        advance = 0
    position = 0
    for iteration in range(upper):
        position += advance
    tl.store(out + (program + position) * B + lane, tl.load(x + lane))
