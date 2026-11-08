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


@triton.jit
def accumulated_matmul(x, y, initial, out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    a = tl.load(x + m[:, None] * K + k[None, :])
    b = tl.load(y + k[:, None] * N + n[None, :])
    c = tl.load(initial + m[:, None] * N + n[None, :])
    value = tl.dot(a, b, c, input_precision="ieee")
    tl.store(out + m[:, None] * N + n[None, :], value)


@triton.jit
def tiled_add(x, y, out, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    i = m[:, None] * N + n[None, :]
    tl.store(out + i, tl.load(x + i) + tl.load(y + i))


@triton.jit
def tiled_matmul(x, y, out, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, K)
    a = tl.load(x + m[:, None] * K + k[None, :])
    b = tl.load(y + k[:, None] * N + n[None, :])
    c = tl.dot(a, b, input_precision="ieee")
    tl.store(out + m[:, None] * N + n[None, :], c)


@triton.jit
def cross_tile_read(out, B: tl.constexpr):
    i = tl.arange(0, B)
    pid = tl.program_id(0)
    value = tl.load(out + (1 - pid) * B + i)
    tl.store(out + pid * B + i, value)


@triton.jit
def loop_matmul(x, y, out, N: tl.constexpr, K: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, K, BK):
        ki = start + k
        a = tl.load(x + m[:, None] * K + ki[None, :])
        b = tl.load(y + ki[:, None] * N + n[None, :])
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(out + m[:, None] * N + n[None, :], acc)


@triton.jit
def loop_stores(x, out, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.store(out + i, tl.load(x + i))
    for start in range(0, 2):
        tl.store(out + i, tl.load(out + i) + start + 1)


@triton.jit
def nested_updates(out, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    for outer in range(0, 2):
        for inner in range(0, 3):
            tl.store(out + i, tl.load(out + i) + 1)
