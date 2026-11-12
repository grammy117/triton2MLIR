"""Update a source view while retaining its complete owning allocation state."""

import triton
import triton.language as tl


@triton.jit
def update_source_view(owner, view, out, rows, cols, s0, s1, steps, bias,
                       BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    active = (row[:, None] < rows) & (col[None, :] < cols)
    address = row[:, None] * s0 + col[None, :] * s1
    for iteration in range(steps):
        value = tl.load(view + address, active, other=0)
        tl.store(view + address, value + bias.to(value.dtype), active)
    value = tl.load(view + address, active, other=0)
    tl.store(out + row[:, None] * cols + col[None, :], value, active)
