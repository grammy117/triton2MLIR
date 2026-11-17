"""Runtime result length is independent of the incoming allocation capacity."""

import triton
import triton.language as tl


@triton.jit
def data_size_prefix(x, n, capacity, out, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + offset, (offset < n) & (offset < capacity), other=0)
    tl.store(out + offset, value, offset < n)
