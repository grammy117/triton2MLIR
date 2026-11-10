"""Uniform gates retain conditional rectangular reads and explicit fills."""

import triton
import triton.language as tl


@triton.jit
def gated_read(x, out, m, n, allow, B: tl.constexpr):
    rows = tl.program_id(0) * B + tl.arange(0, B)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    offsets = rows[:, None] * n + cols[None, :]
    domain = (rows[:, None] < m) & (cols[None, :] < n)
    value = tl.load(x + offsets, domain & allow, other=7.0)
    tl.store(out + offsets, value, domain)
