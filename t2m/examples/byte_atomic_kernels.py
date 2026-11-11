"""Real source byte addresses for indivisible integer word events."""

import triton
import triton.language as tl


@triton.jit
def byte_atomic_addresses(raw, offsets, values, old, n, B: tl.constexpr, WIDTH: tl.constexpr,
                          CAS: tl.constexpr, SCALAR: tl.constexpr):
    lane = tl.program_id(0).to(tl.int64)
    if not SCALAR:
        lane = lane * B + tl.arange(0, B).to(tl.int64)
    at = tl.load(offsets + lane, lane < n, other=0)
    value = tl.load(values + lane, lane < n, other=0)
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    pointer = (raw + at).to(tl.pointer_type(word), bitcast=True)
    if CAS:
        previous = tl.atomic_cas(pointer, value, value ^ 31, sem="acq_rel", scope="gpu")
    else:
        previous = tl.atomic_add(pointer, value, lane < n, sem="relaxed", scope="gpu")
    tl.store(old + lane, previous, lane < n)
