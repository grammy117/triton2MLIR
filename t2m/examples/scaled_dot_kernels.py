"""Packed logical MX tiles; no device launch is needed for TTIR capture."""

import triton
import triton.language as tl


@triton.jit
def scaled_dot_tile(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A_FORMAT: tl.constexpr,
    B_FORMAT: tl.constexpr,
    LHS_K_PACK: tl.constexpr,
    RHS_K_PACK: tl.constexpr,
    HAS_A_SCALE: tl.constexpr,
    HAS_B_SCALE: tl.constexpr,
    GROUP: tl.constexpr,
):
    AM: tl.constexpr = M // 2 if A_FORMAT == "e2m1" and not LHS_K_PACK else M
    AK: tl.constexpr = K // 2 if A_FORMAT == "e2m1" and LHS_K_PACK else K
    BK: tl.constexpr = K // 2 if B_FORMAT == "e2m1" and RHS_K_PACK else K
    BN: tl.constexpr = N // 2 if B_FORMAT == "e2m1" and not RHS_K_PACK else N
    a = tl.load(a_ptr + tl.arange(0, AM)[:, None] * AK + tl.arange(0, AK)[None, :])
    b = tl.load(b_ptr + tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :])
    sa = None
    sb = None
    if HAS_A_SCALE:
        sa = tl.load(
            sa_ptr
            + tl.arange(0, M)[:, None] * (K // GROUP)
            + tl.arange(0, K // GROUP)[None, :]
        )
    if HAS_B_SCALE:
        sb = tl.load(
            sb_ptr
            + tl.arange(0, N)[:, None] * (K // GROUP)
            + tl.arange(0, K // GROUP)[None, :]
        )
    indices = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    c = tl.load(c_ptr + indices)
    result = tl.dot_scaled(
        a,
        sa,
        A_FORMAT,
        b,
        sb,
        B_FORMAT,
        acc=c,
        lhs_k_pack=LHS_K_PACK,
        rhs_k_pack=RHS_K_PACK,
    )
    tl.store(out_ptr + indices, result)


@triton.jit
def scaled_dot_dynamic(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    out_ptr,
    m,
    n,
    k,
    packed_k,
    groups,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    # Allocation extents are ABI data. Their relation to logical K is an
    # executable source assertion, not an assumption from a test's input.
    tl.device_assert(packed_k == tl.cdiv(k, 2), "packed K extent differs")
    tl.device_assert(groups == tl.cdiv(k, 32), "scale group extent differs")
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    packed = tl.arange(0, 16)
    c = tl.load(
        c_ptr + rows[:, None] * n + cols[None, :],
        (rows[:, None] < m) & (cols[None, :] < n),
        other=0.0,
    )
    for step in range(tl.cdiv(k, 32)):
        indices = step * 16 + packed
        a = tl.load(
            a_ptr + rows[:, None] * packed_k + indices[None, :],
            (rows[:, None] < m) & (indices[None, :] < packed_k),
            other=0,
        )
        b = tl.load(
            b_ptr + indices[:, None] * n + cols[None, :],
            (indices[:, None] < packed_k) & (cols[None, :] < n),
            other=0,
        )
        # An odd logical tail leaves an inactive high nibble in a live byte.
        # Keep arbitrary caller bits there from entering the contraction.
        word_mask = tl.where(indices * 2 + 1 < k, 255, 15).to(tl.uint8)
        a = a & word_mask[None, :]
        b = b & word_mask[:, None]
        sa = tl.load(
            sa_ptr + rows[:, None] * groups + step,
            (rows[:, None] < m) & (step < groups),
            other=127,
        )
        sb = tl.load(
            sb_ptr + cols[:, None] * groups + step,
            (cols[:, None] < n) & (step < groups),
            other=127,
        )
        c = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc=c)
    tl.store(
        out_ptr + rows[:, None] * n + cols[None, :],
        c,
        (rows[:, None] < m) & (cols[None, :] < n),
    )
