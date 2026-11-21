"""Logical roots and offsets retain original for/while control and tile values."""

import triton
import triton.language as tl


@triton.jit
def different_root_loop(x,y,out,last,lower,upper,stride,memory_stride,N,B:tl.constexpr):
    lane=tl.program_id(0)*B+tl.arange(0,B)
    pointer=x+(16+lower)*N+lane
    state=tl.load(x+lane,lane<N,other=0)
    for row in range(lower,upper,stride):
        state+=tl.load(pointer,lane<N,other=0)
        if row<3:
            pointer+=memory_stride*N
        else:
            pointer=y+(row-2)*N+lane
    value=tl.load(pointer,lane<N,other=0)
    tl.store(out+lane,state^value,lane<N)
    tl.store(last+lane,value,lane<N)

@triton.jit
def different_root_choice(x,y,out,last,lower,upper,stride,memory_stride,N,B:tl.constexpr,PARTIAL:tl.constexpr=False):
    lane=tl.program_id(0)*B+tl.arange(0,B)
    state=tl.load(x+lane,lane<N,other=0)
    for row in range(lower,upper,stride):
        if row<3:
            pointer=x+(16+row*memory_stride)*N+lane
        else:
            pointer=y+(row-2)*N+lane
        value=tl.load(pointer,lane<N,other=0)
        state+=value
    write=lane<N
    if PARTIAL:
        write=write & (lane%3!=1)
    tl.store(out+lane,state,write)
    tl.store(last+lane,state,write)

@triton.jit
def while_progress(x,y,out,last,lower,upper,stride,memory_stride,N,B:tl.constexpr):
    lane=tl.program_id(0)*B+tl.arange(0,B)
    row=lower
    position=16+lower
    pointer=x+position*N+lane
    state=tl.load(x+lane,lane<N,other=0)
    while row<upper:
        value=tl.load(pointer,(lane<N)&(position>=0)&(position<64),other=0)
        state+=value
        if row<3:
            delta=memory_stride
        else:
            delta=tl.full((),1,memory_stride.dtype)
        position+=delta
        pointer+=delta*N
        row+=stride
    value=tl.load(pointer,(lane<N)&(position>=0)&(position<64),other=0)
    tl.store(out+lane,state^value,lane<N)
    tl.store(last+lane,value,lane<N)
