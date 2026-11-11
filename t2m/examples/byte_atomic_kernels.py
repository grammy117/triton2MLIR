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


@triton.jit
def _byte_atomic_state_step(a, b, pa, pb, out, n, iteration, choice, carry,
                            B: tl.constexpr, WIDTH: tl.constexpr, CAS: tl.constexpr,
                            OP: tl.constexpr, NARROW: tl.constexpr, CARRY: tl.constexpr):
    lane = tl.arange(0, B)
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    pointer = tl.where((lane % 2 == 0) == choice, pa, pb)
    if CAS:
        current = tl.load(pointer)
        comparison = tl.where(lane % 2 == 0, current, current ^ 1)
        desired = carry ^ 31 if CARRY else current ^ 31
        previous = tl.atomic_cas(pointer, comparison, desired, sem="acq_rel", scope="gpu")
    else:
        value = carry if CARRY else (lane + iteration + 3).to(word)
        if OP == "xor": previous = tl.atomic_xor(pointer, value, lane < n, sem="acq_rel", scope="gpu")
        else: previous = tl.atomic_add(pointer, value, lane < n, sem="acq_rel", scope="gpu")
    tl.store(out + iteration * n + lane, previous, lane < n)
    if NARROW:
        tl.store(a.to(tl.pointer_type(tl.int8), bitcast=True) + 1, (iteration + 0x5A).to(tl.int8))
        tl.store(b.to(tl.pointer_type(tl.int8), bitcast=True) + 1, (iteration + 0x33).to(tl.int8))
    advance = tl.where(iteration % 2 == 0, 1, -1)
    carry = previous ^ (lane + iteration + 3).to(word) if CARRY else carry
    return pa + advance, pb + advance, ~choice, carry


@triton.jit
def byte_atomic_state(a, b, out, n, steps, capacity, choice, B: tl.constexpr,
                      WIDTH: tl.constexpr, CAS: tl.constexpr, OP: tl.constexpr,
                      WHILE: tl.constexpr, NARROW: tl.constexpr, CARRY: tl.constexpr):
    lane = tl.arange(0, B).to(tl.int64)
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    pa = (a.to(tl.pointer_type(tl.int8), bitcast=True) + lane * (2 * (WIDTH // 8))).to(tl.pointer_type(word), bitcast=True)
    pb = (b.to(tl.pointer_type(tl.int8), bitcast=True) + lane * (2 * (WIDTH // 8))).to(tl.pointer_type(word), bitcast=True)
    carry = (tl.arange(0, B) + 3).to(word)
    if WHILE:
        iteration = 0
        while iteration < steps:
            pa, pb, choice, carry = _byte_atomic_state_step(a, b, pa, pb, out, n, iteration, choice, carry,
                B, WIDTH, CAS, OP, NARROW, CARRY)
            iteration += 1
    else:
        for iteration in range(steps):
            pa, pb, choice, carry = _byte_atomic_state_step(a, b, pa, pb, out, n, iteration, choice, carry,
                B, WIDTH, CAS, OP, NARROW, CARRY)
