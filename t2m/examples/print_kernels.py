"""Real logical diagnostic kernels; no production dispatch by example name."""

import triton
import triton.language as tl


@triton.jit
def print_integer_words(x, n, steps, allow, B:tl.constexpr, HEX:tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0,B)
    value = tl.load(x + lane,lane < n,other=-7).reshape((2,B // 2))
    for step in range(steps):
        if step % 2 == 0:
            tl.device_print("words",value,value.to(tl.uint64),value.to(tl.uint8),value != 0,hex=HEX)
            tl.device_print("control",steps,allow)
            tl.device_print("event")
    tl.device_assert(allow,"logical print completed")


@triton.jit
def print_float_tiles(x, n, B:tl.constexpr, HEX:tl.constexpr):
    lane = tl.arange(0,B)
    value = tl.load(x + lane,lane < n,other=0.0).reshape((2,B // 2))
    tl.device_print("values",value,hex=HEX)


@triton.jit
def print_narrow_integer_words(x, B:tl.constexpr, HEX:tl.constexpr):
    value = tl.load(x + tl.arange(0,B))
    tl.device_print("narrow",value.to(tl.int8),value.to(tl.int16),value.to(tl.int32),
        value.to(tl.uint16),value.to(tl.uint32),hex=HEX)


@triton.jit
def print_atomic_old(out, allow):
    old = tl.atomic_cas(out,1,3,sem="acq_rel",scope="gpu")
    tl.device_print("old",old)
    tl.device_assert(allow,"atomic print completed")
