"""A real pointer update whose scalar argument is captured before invocation."""

import triton
import triton.language as tl


@triton.jit
def scalar_carrier_update(state, step, N: tl.constexpr):
    indices = tl.arange(0,N)
    value = tl.load(state+indices)
    tl.store(state+indices,value+step)
