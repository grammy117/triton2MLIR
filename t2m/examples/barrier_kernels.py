"""Synchronous tile exchange with observable cross-lane communication."""

import triton
import triton.language as tl


@triton.jit
def exchange_tile(state, out, n, steps, B: tl.constexpr, CROSS: tl.constexpr = False):
    origin = tl.program_id(0) * B
    local = tl.arange(0, B)
    row = origin + local[:, None]
    column = local[None, :]
    pointer = row * B + column
    tl.store(out + pointer, tl.full((B, B), -7, tl.int64), row < n)
    for step in range(steps):
        before = tl.load(state + pointer, row < n, other=0)
        tl.store(state + pointer, before + step.to(tl.int64) + 1, row < n)
        tl.debug_barrier()
        # Read another lane's newly written value, within this program's tile.
        transposed_row = origin + local[None, :]
        if CROSS:
            # A negative fixture: a CTA barrier cannot order another program.
            transposed_row += B
        transposed_column = local[:, None]
        after = tl.load(state + transposed_row * B + transposed_column,
                        transposed_row < n, other=-3)
        tl.store(out + pointer, after, row < n)
        # Complete all reads before the next iteration overwrites the tile.
        tl.debug_barrier()
