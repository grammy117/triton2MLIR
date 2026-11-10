"""Runtime signed strides into explicit source allocations, preserving tiles."""

import triton
import triton.language as tl


@triton.jit
def strided_tiled_dot(x, y, initial, out, m, n, k, a_row, a_k, b_k, b_col,
                     a_base, b_base, a_count, b_count,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    state = tl.load(initial + rows[:, None]*n + cols[None, :],
                    (rows[:, None] < m) & (cols[None, :] < n), other=0.)
    for step in range(tl.cdiv(k, BK)):
        indices = step*BK + inner
        a = tl.load(x + a_base + rows[:, None]*a_row + indices[None, :]*a_k,
                    (rows[:, None] < m) & (indices[None, :] < k), other=0.)
        b = tl.load(y + b_base + indices[:, None]*b_k + cols[None, :]*b_col,
                    (indices[:, None] < k) & (cols[None, :] < n), other=0.)
        state = tl.dot(a,b,state,input_precision="ieee",max_num_imprecise_acc=0)
    tl.store(out + rows[:, None]*n + cols[None, :], state,
             (rows[:, None] < m) & (cols[None, :] < n))


@triton.jit
def strided_output_dot(x, y, initial, out, m, n, k, a_row, a_k, b_k, b_col,
                       a_base, b_base, a_count, b_count,
                       out_row, out_col, out_base, out_count,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0,BM)
    cols = tl.program_id(1)*BN + tl.arange(0,BN)
    inner = tl.arange(0,BK)
    state = tl.load(initial + rows[:,None]*n + cols[None,:],
                    (rows[:,None] < m) & (cols[None,:] < n), other=0.)
    for step in range(tl.cdiv(k,BK)):
        indices = step*BK + inner
        a = tl.load(x + a_base + rows[:,None]*a_row + indices[None,:]*a_k,
                    (rows[:,None] < m) & (indices[None,:] < k), other=0.)
        b = tl.load(y + b_base + indices[:,None]*b_k + cols[None,:]*b_col,
                    (indices[:,None] < k) & (cols[None,:] < n), other=0.)
        state = tl.dot(a,b,state,input_precision="ieee",max_num_imprecise_acc=0)
    tl.store(out + out_base + rows[:,None]*out_row + cols[None,:]*out_col, state,
             (rows[:,None] < m) & (cols[None,:] < n))
