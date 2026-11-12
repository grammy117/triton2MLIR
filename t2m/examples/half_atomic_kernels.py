"""Half atomic additions retain original storage and exact integer old words."""

import triton
import triton.language as tl


@triton.jit
def half_atomic_add(x, index, state, old, n, m, B: tl.constexpr, SCALAR: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    active = (lane < n) & (lane % 3 != 1)
    destination = tl.load(index + lane, lane < n, other=-8191)
    value = tl.load(x + lane, lane < n, other=0)
    previous = tl.atomic_add(
        state + destination, value, active, sem="acq_rel", scope="sys"
    )
    tl.store(old + lane, previous.to(tl.uint16, bitcast=True), lane < n)
