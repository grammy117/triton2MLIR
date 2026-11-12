"""Tiled contractions over source allocation views with FP8 residual casts."""

import triton
import triton.language as tl


@triton.jit
def residual_dot(left, right, state, PRECISION: tl.constexpr):
    left = left + left.to(tl.float8e5, fp_downcast_rounding="rtne").to(tl.float32)
    right = right + right.to(tl.float8e4nv, fp_downcast_rounding="rtz").to(tl.float32)
    state = state + state.to(tl.float8e5, fp_downcast_rounding="rtz").to(tl.float32)
    return tl.dot(
        left, right, state, input_precision=PRECISION, max_num_imprecise_acc=0
    )


@triton.jit
def fp8_allocation_dot(
    x,
    y,
    initial,
    out,
    m,
    n,
    k,
    a_row,
    a_k,
    b_k,
    b_col,
    a_base,
    b_base,
    a_count,
    b_count,
    out_row,
    out_col,
    out_base,
    out_count,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PRECISION: tl.constexpr,
    DESCRIPTORS: tl.constexpr,
):
    row = tl.program_id(0) * BM
    col = tl.program_id(1) * BN
    rows = row + tl.arange(0, BM)
    cols = col + tl.arange(0, BN)
    state = tl.load(
        initial + rows[:, None] * n + cols[None, :],
        (rows[:, None] < m) & (cols[None, :] < n),
        other=0.0,
    )
    for step in range(tl.cdiv(k, BK)):
        if DESCRIPTORS:
            left = x.load([row, step * BK])
            right = y.load([step * BK, col])
        else:
            inner = step * BK + tl.arange(0, BK)
            left = tl.load(
                x + a_base + rows[:, None] * a_row + inner[None, :] * a_k,
                (rows[:, None] < m) & (inner[None, :] < k),
                other=0.0,
            )
            right = tl.load(
                y + b_base + inner[:, None] * b_k + cols[None, :] * b_col,
                (inner[:, None] < k) & (cols[None, :] < n),
                other=0.0,
            )
        state = residual_dot(left, right, state, PRECISION)
    tl.store(
        out + out_base + rows[:, None] * out_row + cols[None, :] * out_col,
        state,
        (rows[:, None] < m) & (cols[None, :] < n),
    )
