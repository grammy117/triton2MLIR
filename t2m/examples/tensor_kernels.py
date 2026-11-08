"""Small source kernels exercising the reusable tensor-compute pipeline."""

import triton
import triton.language as tl


@triton.jit
def matrix_add(x, y, out, M: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, M)[:, None]
    col = tl.arange(0, N)[None, :]
    offset = row * N + col
    tl.store(out + offset, tl.load(x + offset) + tl.load(y + offset))


@triton.jit
def row_softmax(x, out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    value = tl.load(x + i, i < N, other=-float("inf"))
    e = tl.exp(value - tl.max(value, axis=0))
    tl.store(out + i, e / tl.sum(e, axis=0), i < N)


@triton.jit
def block_matmul(x, y, out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    a = tl.load(x + m[:, None] * K + k[None, :])
    b = tl.load(y + k[:, None] * N + n[None, :])
    c = tl.dot(a, b, input_precision="ieee")
    tl.store(out + m[:, None] * N + n[None, :], c)


@triton.jit
def sequenced_outputs(x, out, second, N: tl.constexpr):
    i = tl.arange(0, N)
    value = tl.load(x + i)
    tl.store(out + i, value + 1)
    updated = tl.load(out + i)
    tl.store(out + i, updated * 2)
    tl.store(second + i, updated - 3)
