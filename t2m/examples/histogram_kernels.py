"""Per-program histograms retain logical tiles and exact source masks."""

import triton
import triton.language as tl


@triton.jit
def histogram_tiles(x, out, n, tiles, B: tl.constexpr, NB: tl.constexpr, MASKED: tl.constexpr, ROW_VIEW: tl.constexpr):
    program = tl.program_id(0)
    lane = program * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=0)
    if MASKED:
        counts = tl.histogram(value, NB, mask=(lane < n) & (lane % 3 != 1))
    else:
        counts = tl.histogram(value, NB)
    if ROW_VIEW:
        rows = program + tl.arange(0, 1)[:, None]
        columns = tl.arange(0, NB)[None, :]
        tl.store(out + rows * NB + columns, counts[None, :], rows < tiles)
    else:
        tl.store(out + program * NB + tl.arange(0, NB), counts)
