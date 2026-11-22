"""Single-head pre-norm transformer block with fused, register-resident reuse.

Inference: no dropout, full bidirectional attention, f32/IEEE dot products.
One logical program owns a sequence. This source intentionally uses ordinary
Triton tensor computation, with no assembly or hardware-specific instructions.
"""

import triton
import triton.language as tl


@triton.jit
def _layer_norm(x, gamma, beta, D: tl.constexpr, EPS: tl.constexpr):
    mean = tl.sum(x, 1) / D
    centered = x - mean[:, None]
    variance = tl.sum(centered * centered, 1) / D
    return centered * tl.rsqrt(variance[:, None] + EPS) * gamma[None, :] + beta[None, :]


@triton.jit
def transformer_block(x, wq, wk, wv, wo, w1, w2, g1, b1, g2, b2, out,
                      S: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                      BS: tl.constexpr, EPS: tl.constexpr):
    row = tl.arange(0, BS)
    dim = tl.arange(0, D)
    hidden = tl.arange(0, H)
    value = tl.load(x + row[:, None] * D + dim[None, :], row[:, None] < S, other=0)
    gamma1 = tl.load(g1 + dim)
    beta1 = tl.load(b1 + dim)
    norm = _layer_norm(value, gamma1, beta1, D, EPS)
    square = dim[:, None] * D + dim[None, :]
    query = tl.dot(norm, tl.load(wq + square), input_precision="ieee")
    key = tl.dot(norm, tl.load(wk + square), input_precision="ieee")
    val = tl.dot(norm, tl.load(wv + square), input_precision="ieee")
    scores = tl.dot(query, tl.trans(key), input_precision="ieee") * (D ** -0.5)
    scores = tl.where(row[None, :] < S, scores, -float("inf"))
    probability = tl.exp(scores - tl.max(scores, 1)[:, None])
    probability = probability / tl.sum(probability, 1)[:, None]
    attended = tl.dot(probability, val, input_precision="ieee")
    residual = value + tl.dot(attended, tl.load(wo + square), input_precision="ieee")
    gamma2 = tl.load(g2 + dim)
    beta2 = tl.load(b2 + dim)
    norm2 = _layer_norm(residual, gamma2, beta2, D, EPS)
    up = tl.load(w1 + dim[:, None] * H + hidden[None, :])
    down = tl.load(w2 + hidden[:, None] * D + dim[None, :])
    activation = tl.maximum(tl.dot(norm2, up, input_precision="ieee"), 0)
    answer = residual + tl.dot(activation, down, input_precision="ieee")
    tl.store(out + row[:, None] * D + dim[None, :], answer, row[:, None] < S)
