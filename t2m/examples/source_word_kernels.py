"""Kernels whose source integer words and pointer recurrences are observable."""

import triton
import triton.language as tl


@triton.jit
def modular_update(x, encoded, out, n, m, B: tl.constexpr, SCALAR: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    key = tl.load(encoded + lane, lane < n, other=0)
    index = key * 4 + 1
    previous = tl.load(out + index, lane < n, other=0)
    value = tl.load(x + lane, lane < n, other=0)
    tl.store(out + index, previous + value + 1, lane < n)


@triton.jit
def pointer_cancel(x, encoded, out, n, m, B: tl.constexpr, SCALAR: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    key = tl.load(encoded + lane, lane < n, other=0)
    pointer = out + lane
    pointer += key
    pointer += key
    pointer += -2 * key.to(tl.int64)
    value = tl.load(x + lane, lane < n, other=0)
    tl.store(pointer, value + 1, lane < n)


@triton.jit
def pointer_runtime_sum(x, out, n, rows, steps, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    pointer = x + lane
    acc = tl.full((B,), 0, tl.float32)
    for step in range(steps):
        acc += tl.load(pointer, lane < n, other=0)
        pointer += n
    acc += tl.load(pointer, lane < n, other=0)
    tl.store(out + lane, acc, lane < n)


@triton.jit
def permuted_tiles(x, out, N: tl.constexpr, B: tl.constexpr,
                    PROGRAMS: tl.constexpr, STEP: tl.constexpr):
    tile = (tl.program_id(0) * STEP + 1) % PROGRAMS
    lane = tile * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < N, other=0)
    tl.store(out + lane, value + 1, lane < N)
