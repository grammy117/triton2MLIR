"""Real source byte addresses for indivisible integer word events."""

import triton
import triton.language as tl


@triton.jit
def byte_atomic_addresses(raw, offsets, values, old, n, B: tl.constexpr, WIDTH: tl.constexpr,
                          CAS: tl.constexpr, SCALAR: tl.constexpr):
    lane = tl.program_id(0).to(tl.int64)
    if not SCALAR:
        lane = lane * B + tl.arange(0, B).to(tl.int64)
    at = tl.load(offsets + lane, lane < n, other=-9223372036854775807)
    value = tl.load(values + lane, lane < n, other=0)
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    pointer = (raw + at).to(tl.pointer_type(word), bitcast=True)
    if CAS:
        previous = tl.atomic_cas(pointer, value, value ^ 31, sem="acq_rel", scope="gpu")
    else:
        previous = tl.atomic_add(pointer, value, lane < n, sem="relaxed", scope="gpu")
    tl.store(old + lane, previous, lane < n)


@triton.jit
def byte_atomic_rmw(raw, offsets, values, old, n, B: tl.constexpr, WIDTH: tl.constexpr,
                    OP: tl.constexpr, SCALAR: tl.constexpr):
    lane = tl.program_id(0).to(tl.int64)
    if not SCALAR:
        lane = lane * B + tl.arange(0, B).to(tl.int64)
    at = tl.load(offsets + lane, lane < n, other=-9223372036854775807)
    value = tl.load(values + lane, lane < n, other=0)
    word: tl.constexpr = tl.int32 if WIDTH == 32 else tl.int64
    pointer = (raw + at).to(tl.pointer_type(word), bitcast=True)
    mask = lane < n
    if OP == "add": previous = tl.atomic_add(pointer, value, mask, sem="relaxed", scope="gpu")
    elif OP == "and": previous = tl.atomic_and(pointer, value, mask, sem="relaxed", scope="gpu")
    elif OP == "or": previous = tl.atomic_or(pointer, value, mask, sem="relaxed", scope="gpu")
    elif OP == "xor": previous = tl.atomic_xor(pointer, value, mask, sem="relaxed", scope="gpu")
    elif OP == "max": previous = tl.atomic_max(pointer, value, mask, sem="relaxed", scope="gpu")
    elif OP == "min": previous = tl.atomic_min(pointer, value, mask, sem="relaxed", scope="gpu")
    elif OP == "exch": previous = tl.atomic_xchg(pointer, value, mask, sem="relaxed", scope="gpu")
    else:
        unsigned: tl.constexpr = tl.uint32 if WIDTH == 32 else tl.uint64
        pointer = pointer.to(tl.pointer_type(unsigned), bitcast=True)
        value = value.to(unsigned)
        if OP == "umax": previous = tl.atomic_max(pointer, value, mask, sem="relaxed", scope="gpu")
        else: previous = tl.atomic_min(pointer, value, mask, sem="relaxed", scope="gpu")
    tl.store(old + lane, previous, mask)


@triton.jit
def byte_atomic_conflict(raw, out, REVERSE: tl.constexpr):
    pid = tl.program_id(0)
    if (pid == 0) != REVERSE:
        pointer = raw.to(tl.pointer_type(tl.int32), bitcast=True)
        value = tl.atomic_add(pointer, 1, sem="acq_rel", scope="gpu")
    else:
        # The final byte, not just the first byte, belongs to the word event.
        value = tl.load(raw + 3).to(tl.int32)
    tl.store(out + pid, value)
