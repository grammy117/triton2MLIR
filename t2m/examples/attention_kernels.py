"""Tiled online softmax with running maximum, normalizer and accumulator."""

import triton
import triton.language as tl


@triton.jit
def online_attention(q, k, v, out, m, n, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.arange(0, BN)
    channel = tl.arange(0, D)
    query = tl.load(q + row[:, None] * D + channel[None, :], row[:, None] < m, other=0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    normalization = tl.full((BM,), 0, tl.float32)
    accumulator = tl.full((BM, D), 0, tl.float32)
    for start in range(0, n, BN):
        key_row = start + col
        key = tl.load(k + key_row[:, None] * D + channel[None, :], key_row[:, None] < n, other=0)
        value = tl.load(v + key_row[:, None] * D + channel[None, :], key_row[:, None] < n, other=0)
        scores = tl.dot(query, tl.trans(key), input_precision="ieee") * (D ** -0.5)
        scores = tl.where(key_row[None, :] < n, scores, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp(maximum - new_maximum)
        probability = tl.exp(scores - new_maximum[:, None])
        normalization = normalization * alpha + tl.sum(probability, 1)
        accumulator = tl.dot(probability, value, accumulator * alpha[:, None], input_precision="ieee")
        maximum = new_maximum
    # An explicit source convention, including the zero-key case.
    divisor = tl.where(normalization > 0, normalization, 1)
    answer = accumulator / divisor[:, None]
    tl.store(out + row[:, None] * D + channel[None, :], answer, row[:, None] < m)
