"""Public source casts with explicit downcast rounding, above device mapping."""

import triton
import triton.language as tl


@triton.jit
def rounded_cast(x, out, n, B: tl.constexpr, ROUND: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=0)
    result = value.to(out.dtype.element_ty, fp_downcast_rounding=ROUND)
    tl.store(out + lane, result, lane < n)


@triton.jit
def rounded_cast_steps(x, out, n, steps, delta, B: tl.constexpr, ROUND: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=0)
    for step in range(steps):
        value = (
            (value + delta)
            .to(out.dtype.element_ty, fp_downcast_rounding=ROUND)
            .to(tl.float32)
        )
    result = value.to(out.dtype.element_ty, fp_downcast_rounding=ROUND)
    tl.store(out + lane, result, lane < n)
