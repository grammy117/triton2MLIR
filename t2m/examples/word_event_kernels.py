"""Region-local memory with exact nonuniform fills and source recurrences."""

import triton
import triton.language as tl

from .tensor_kernels import loop_matmul


@triton.jit
def conditional_events(x, fill, out, n, steps, choose, shift,
                       B: tl.constexpr, SCALAR: tl.constexpr, WHILE: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    if choose:
        if WHILE:
            i = 0
            while i < steps:
                other = tl.load(fill + lane, lane < n, other=0)
                if i % 2 == 0:
                    value = tl.load(x + lane + shift, (lane < n) & (lane % 2 == 0), other=other)
                else:
                    value = tl.load(out + lane, lane < n, other=other)
                tl.store(out + lane, value, lane < n)
                i += 1
        else:
            for i in range(steps):
                other = tl.load(fill + lane, lane < n, other=0)
                if i % 2 == 0:
                    value = tl.load(x + lane + shift, (lane < n) & (lane % 2 == 0), other=other)
                else:
                    value = tl.load(out + lane, lane < n, other=other)
                tl.store(out + lane, value, lane < n)


@triton.jit
def branch_matmul(x, y, out, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    if tl.program_id(0) % 2 == 0:
        loop_matmul(x, y, out, N, K, BM, BN, BK)
    else:
        loop_matmul(x, y, out, N, K, BM, BN, BK)
