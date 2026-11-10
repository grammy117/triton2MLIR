"""Logical FP8 contractions: tiles, source accumulator and runtime K remain."""

import triton
import triton.language as tl


@triton.jit
def fp8_tiled_dot(
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
    zero_a = tl.full((), 0, tl.uint8).to(x.dtype.element_ty, bitcast=True)
    zero_b = tl.full((), 0, tl.uint8).to(y.dtype.element_ty, bitcast=True)
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
            other=zero_a,
        )
        b = tl.load(
            y + indices[:, None] * n + cols[None, :],
            (indices[:, None] < k) & (cols[None, :] < n),
            other=zero_b,
        )
        c = tl.dot(a, b, c, input_precision=PRECISION, max_num_imprecise_acc=0)
    tl.store(
        out + rows[:, None] * n + cols[None, :],
        c,
        (rows[:, None] < m) & (cols[None, :] < n),
    )


@triton.jit
def fp8_negative_zero_dot(
    x, y, out, bonus, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    inner = tl.arange(0, K)
    a = tl.load(x + rows[:, None] * K + inner[None, :])
    b = tl.load(y + inner[:, None] * N + cols[None, :])
    # The pinned AST frontend folds a literal -0.0 to +0.0. An explicit
    # integer word bitcast expresses the intended source negative zero.
    c = tl.full((M, N), -2147483648, tl.int32).to(tl.float32, bitcast=True)
    result = tl.dot(a, b, c, input_precision="ieee", max_num_imprecise_acc=0)
    tl.store(out + rows[:, None] * N + cols[None, :], result + bonus)
