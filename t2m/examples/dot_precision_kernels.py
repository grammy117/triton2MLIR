"""Source precision modes with logical tiles, runtime tails and original C."""

import triton
import triton.language as tl


@triton.jit
def precision_tiled_dot(
    x,
    y,
    initial,
    out,
    m,
    n,
    k,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PRECISION: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    c = tl.load(
        initial + rows[:, None] * n + cols[None, :],
        (rows[:, None] < m) & (cols[None, :] < n),
        other=0.0,
    )
    for step in range(tl.cdiv(k, BK)):
        indices = step * BK + inner
        a = tl.load(
            x + rows[:, None] * k + indices[None, :],
            (rows[:, None] < m) & (indices[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            y + indices[:, None] * n + cols[None, :],
            (indices[:, None] < k) & (cols[None, :] < n),
            other=0.0,
        )
        c = tl.dot(a, b, c, input_precision=PRECISION, max_num_imprecise_acc=0)
    tl.store(
        out + rows[:, None] * n + cols[None, :],
        c,
        (rows[:, None] < m) & (cols[None, :] < n),
    )
