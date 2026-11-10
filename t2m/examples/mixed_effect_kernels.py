"""Real kernels combining tensor results with source diagnostic effects."""

import triton
import triton.language as tl


@triton.jit
def update_and_print(x,state,out,n,steps,allow,B:tl.constexpr):
    lane = tl.program_id(0)*B+tl.arange(0,B)
    active = lane < n
    value = tl.load(x+lane,active,other=0)
    current = tl.load(state+lane,active,other=0)
    for step in range(steps):
        if step % 2 == 0:
            current = current+value
            tl.device_print("state",current.reshape((2,B//2)))
    tl.device_assert(allow,"mixed result check")
    tl.store(state+lane,current,active)
    tl.store(out+lane,value-current,active)


@triton.jit
def copy_and_print(x,out,n,B:tl.constexpr):
    lane = tl.program_id(0)*B+tl.arange(0,B)
    value = tl.load(x+lane,lane < n,other=0)
    tl.device_print("copy",value,hex=True)
    tl.store(out+lane,value,lane < n)


@triton.jit
def copy_scale_and_print(x,out,scaled,n,B:tl.constexpr):
    lane = tl.program_id(0)*B+tl.arange(0,B)
    value = tl.load(x+lane,lane < n,other=0)
    twice = value*2.0
    tl.device_print("scaled",twice)
    tl.store(out+lane,value,lane < n)
    tl.store(scaled+lane,twice,lane < n)
