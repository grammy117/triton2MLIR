"""Logical, two-axis tile permutation without a physical lane mapping."""

import triton
import triton.language as tl


@triton.jit
def tile_gather_values(x, indices, out, AXIS: tl.constexpr, R: tl.constexpr, S: tl.constexpr,
                       OR: tl.constexpr, OS: tl.constexpr):
    row, col = tl.arange(0, R)[:, None], tl.arange(0, S)[None, :]
    source = tl.load(x + row * S + col)
    out_row, out_col = tl.arange(0, OR)[:, None], tl.arange(0, OS)[None, :]
    selected = tl.load(indices + out_row * OS + out_col)
    value = tl.gather(source, selected, AXIS)
    tl.store(out + out_row * OS + out_col, value)
