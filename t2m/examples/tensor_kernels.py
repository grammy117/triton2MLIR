"""Small source kernels exercising the reusable tensor-compute pipeline."""

import triton
import triton.language as tl


@triton.jit
def blockptr_add(x, out, N: tl.constexpr, B: tl.constexpr):
    start = tl.program_id(0) * B
    src = tl.make_block_ptr(x, shape=(N,), strides=(1,), offsets=(start,), block_shape=(B,), order=(0,))
    dst = tl.make_block_ptr(out, shape=(N,), strides=(1,), offsets=(start,), block_shape=(B,), order=(0,))
    value = tl.load(src, boundary_check=(0,), padding_option="zero")
    tl.store(dst, value + 1, boundary_check=(0,))


@triton.jit
def blockptr_matmul(x, y, out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m, n = tl.program_id(0) * BM, tl.program_id(1) * BN
    a = tl.make_block_ptr(x, shape=(M, K), strides=(K, 1), offsets=(m, 0), block_shape=(BM, BK), order=(1, 0))
    b = tl.make_block_ptr(y, shape=(K, N), strides=(N, 1), offsets=(0, n), block_shape=(BK, BN), order=(1, 0))
    c = tl.make_block_ptr(out, shape=(M, N), strides=(N, 1), offsets=(m, n), block_shape=(BM, BN), order=(1, 0))
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        av = tl.load(a, boundary_check=(0, 1), padding_option="zero")
        bv = tl.load(b, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(av, bv, acc, input_precision="ieee")
        a = tl.advance(a, (0, BK))
        b = tl.advance(b, (BK, 0))
    tl.store(c, acc, boundary_check=(0, 1))


@triton.jit
def descriptor_add(x, out, N: tl.constexpr, B: tl.constexpr):
    src = tl.make_tensor_descriptor(x, shape=[N], strides=[1], block_shape=[B])
    dst = tl.make_tensor_descriptor(out, shape=[N], strides=[1], block_shape=[B])
    start = tl.program_id(0) * B
    value = src.load([start])
    dst.store([start], value + 1)


@triton.jit
def descriptor_matmul(x, y, out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    a = tl.make_tensor_descriptor(x, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b = tl.make_tensor_descriptor(y, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    c = tl.make_tensor_descriptor(out, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    m, n = tl.program_id(0) * BM, tl.program_id(1) * BN
    acc = tl.full((BM, BN), 0, tl.float32)
    for k in range(0, K, BK):
        av = a.load([m, k])
        bv = b.load([k, n])
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    c.store([m, n], acc)


@triton.jit
def _scan_maximum(left, right):
    return tl.maximum(left, right)


@triton.jit
def scan_tiles(x, out, N: tl.constexpr, B: tl.constexpr, KIND: tl.constexpr, REVERSE: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    if KIND == 0:
        value = tl.load(x + i, i < N, other=0)
        result = tl.cumsum(value, axis=0, reverse=REVERSE)
    elif KIND == 1:
        value = tl.load(x + i, i < N, other=1)
        result = tl.cumprod(value, axis=0, reverse=REVERSE)
    else:
        fill = -float("inf") if x.dtype.element_ty.is_floating() else -2147483648
        value = tl.load(x + i, i < N, other=fill)
        result = tl.associative_scan(value, 0, _scan_maximum, reverse=REVERSE)
    tl.store(out + i, result, i < N)


@triton.jit
def _compose_affine(left_a, left_b, right_a, right_b):
    return right_a * left_a, right_a * left_b + right_b


@triton.jit
def tuple_scan(a, b, out_a, out_b, R: tl.constexpr, S: tl.constexpr, AXIS: tl.constexpr, REVERSE: tl.constexpr):
    i = tl.arange(0, R)[:, None] * S + tl.arange(0, S)[None, :]
    av, bv = tl.load(a + i), tl.load(b + i)
    result_a, result_b = tl.associative_scan((av, bv), AXIS, _compose_affine, reverse=REVERSE)
    tl.store(out_a + i, result_a)
    tl.store(out_b + i, result_b)


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
def ragged_add(x, out, N: tl.constexpr, LIMIT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + i, i < N, other=7.0)
    tl.store(out + i, value * 2, i < LIMIT)


@triton.jit
def ragged_matmul(x, y, out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TRANSPOSED: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, K, BK):
        kk = start + k
        a = tl.load(x + m[:, None] * K + kk[None, :],
                    (m[:, None] < M) & (kk[None, :] < K), other=0)
        if TRANSPOSED:
            b = tl.load(y + n[None, :] * K + kk[:, None],
                        (kk[:, None] < K) & (n[None, :] < N), other=0)
        else:
            b = tl.load(y + kk[:, None] * N + n[None, :],
                        (kk[:, None] < K) & (n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(out + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def ragged_fill_reduce(x, out, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + i, i < N, other=7.0)
    total = tl.sum(value, axis=0)
    if total > 0:
        tl.store(out + i, value + total, i < N)
    else:
        tl.store(out + i, value - total, i < N)


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


@triton.jit
def transposed_loop_matmul(x, weight, out, N: tl.constexpr, K: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, K, BK):
        ki = start + k
        a = tl.load(x + m[:, None] * K + ki[None, :])
        b = tl.load(weight + n[None, :] * K + ki[:, None])
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(out + m[:, None] * N + n[None, :], acc)


@triton.jit
def transpose_store(x, out, M: tl.constexpr, N: tl.constexpr):
    m = tl.arange(0, M)[:, None]
    n = tl.arange(0, N)[None, :]
    value = tl.load(x + m * N + n)
    tl.store(out + n * M + m, value)


@triton.jit
def conditional_update(x, out, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + i)
    if tl.sum(value, axis=0) > 0:
        tl.store(out + i, value + 3)
    else:
        tl.store(out + i, value * 2)


@triton.jit
def conditional_partial(x, out, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + i)
    if tl.sum(value, axis=0) > 0:
        tl.store(out + i, value + 3)


@triton.jit
def conditional_uninitialized(x, out, B: tl.constexpr):
    i = tl.arange(0, B)
    value = tl.load(x + i)
    if tl.sum(value, axis=0) > 0:
        tl.store(out + i, value)
    else:
        tl.store(out + i, tl.load(out + i) + value)


@triton.jit
def loop_conditional_update(x, out, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + i)
    tl.store(out + i, value)
    for step in range(0, 3):
        previous = tl.load(out + i)
        if tl.sum(value, axis=0) > 0:
            chosen = previous + 1
            tl.store(out + i, chosen)
        else:
            chosen = previous * 2
            tl.store(out + i, chosen)
        tl.store(out + i, chosen + 1)


@triton.jit
def pointer_loop_matmul(x, y, out, N: tl.constexpr, K: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    ap = x + m[:, None] * K + k[None, :]
    bp = y + k[:, None] * N + n[None, :]
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(0, K // BK):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc, input_precision="ieee")
        ap += BK
        bp += BK * N
    tl.store(out + m[:, None] * N + n[None, :], acc)


@triton.jit
def pointer_loop_after(x, out, B: tl.constexpr):
    i = tl.arange(0, B)
    pointer = x + i
    acc = tl.full((B,), 0, tl.float32)
    for step in range(2, 8, 2):
        acc += tl.load(pointer)
        pointer += B
    tl.store(out + i, acc + tl.load(pointer))
