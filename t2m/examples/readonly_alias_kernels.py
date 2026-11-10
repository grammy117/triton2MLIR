"""A real pointer update whose scalar argument is captured before invocation."""

import triton
import triton.language as tl


@triton.jit
def scalar_carrier_update(state, step, N: tl.constexpr):
    indices = tl.arange(0,N)
    value = tl.load(state+indices)
    tl.store(state+indices,value+step)


@triton.jit
def aliased_updates(read, first, second, n, steps, B: tl.constexpr):
    indices = tl.program_id(0)*B+tl.arange(0,B)
    active = indices < n
    for iteration in range(steps):
        value = tl.load(read+indices,active,other=0)
        tl.store(first+indices,value+3,active)
        forwarded = tl.load(read+indices,active,other=0)
        tl.store(second+indices,forwarded*2,active)
