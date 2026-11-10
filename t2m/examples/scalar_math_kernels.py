"""Real public Triton precise math and NaN-policy examples."""

import triton
import triton.language as tl


@triton.jit
def precise_values(x,y,low,high,no_nan,all_nan,divided,root,n,B:tl.constexpr):
    lane = tl.program_id(0)*B+tl.arange(0,B)
    active = lane < n
    a = tl.load(x+lane,active,other=0)
    b = tl.load(y+lane,active,other=1)
    lo = tl.load(low+lane,active,other=-1)
    hi = tl.load(high+lane,active,other=1)
    tl.store(no_nan+lane,tl.clamp(a,lo,hi,propagate_nan=tl.PropagateNan.NONE),active)
    tl.store(all_nan+lane,tl.clamp(a,lo,hi,propagate_nan=tl.PropagateNan.ALL),active)
    tl.store(divided+lane,tl.math.div_rn(a,b),active)
    tl.store(root+lane,tl.math.sqrt_rn(a),active)
