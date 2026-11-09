"""Unmasked logical rows and partial updates with runtime launch domains."""

import triton
import triton.language as tl


@triton.jit
def row_cells(x, out, n, launch, B: tl.constexpr, STEP: tl.constexpr, SHIFT: tl.constexpr,
              CONDITIONAL: tl.constexpr):
    program = tl.program_id(0)
    row = program * STEP + tl.arange(0, 1)[:, None]
    column = tl.arange(0, B)[None, :]
    value = tl.load(x + row * B + column)
    if CONDITIONAL:
        if program % 2 == 0:
            tl.store(out + (row + SHIFT) * B + column, value + 1)
    else:
        tl.store(out + (row + SHIFT) * B + column, value + 1)


@triton.jit
def row_tail_matrix(x, out, m, n, B: tl.constexpr, SHIFT: tl.constexpr):
    row = tl.program_id(0) + SHIFT + tl.arange(0, 1)[:, None]
    column = tl.program_id(1) * B + tl.arange(0, B)[None, :]
    value = tl.load(x + row * n + column, column < n, other=0)
    tl.store(out + row * n + column, value + 1, column < n)


@triton.jit
def flat_unmasked_tiles(x, out, n, launch, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane)
    tl.store(out + lane, value + 1)
