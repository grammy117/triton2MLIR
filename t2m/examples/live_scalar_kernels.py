"""Runtime values passed by the model retain their actual typed scalar edges."""

import triton
import triton.language as tl


@triton.jit
def live_scalar_values(x, control, out, N: tl.constexpr, B: tl.constexpr, KIND: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < N, other=0)
    if KIND == "bool":
        result = tl.where(control, value, -value)
    elif KIND == "float":
        result = value + control
    else:
        result = value ^ control
    tl.store(out + lane, result, lane < N)


@triton.jit
def live_extent_copy(x, n, out, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=0)
    tl.store(out + lane, value, lane < n)


@triton.jit
def live_extent_weighted(x, weight, n, out, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=0)
    control = tl.load(weight)
    tl.store(out + lane, value ^ control, lane < n)
