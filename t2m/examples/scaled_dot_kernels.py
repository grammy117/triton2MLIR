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
