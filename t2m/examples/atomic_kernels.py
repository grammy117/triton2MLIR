"""Indexed integer atomics with original old values and ordinary observations."""

import triton
import triton.language as tl


@triton.jit
def indexed_atomic(x, index, state, old, n, m, B: tl.constexpr, OP: tl.constexpr,
                   SCALAR: tl.constexpr, MASK: tl.constexpr, FRESH: tl.constexpr,
                   SEM: tl.constexpr, SCOPE: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    at = tl.load(index + lane, lane < n, other=-123)
    value = tl.load(x + lane, lane < n, other=0)
    if FRESH:
        tl.store(state + lane, 0, lane < n)
    active = lane < n
    if MASK:
        active = active & (lane % 3 != 1)
    if OP == "add":
        previous = tl.atomic_add(state + at, value, active, sem=SEM, scope=SCOPE)
    elif OP == "and":
        previous = tl.atomic_and(state + at, value, active, sem=SEM, scope=SCOPE)
    elif OP == "or":
        previous = tl.atomic_or(state + at, value, active, sem=SEM, scope=SCOPE)
    elif OP == "xor":
        previous = tl.atomic_xor(state + at, value, active, sem=SEM, scope=SCOPE)
    elif OP == "max" or OP == "umax":
        previous = tl.atomic_max(state + at, value, active, sem=SEM, scope=SCOPE)
    elif OP == "min" or OP == "umin":
        previous = tl.atomic_min(state + at, value, active, sem=SEM, scope=SCOPE)
    else:
        previous = tl.atomic_xchg(state + at, value, active, sem=SEM, scope=SCOPE)
    tl.store(old + lane, previous, lane < n)


@triton.jit
def mixed_atomic_observation(state, out, REVERSE: tl.constexpr):
    program = tl.program_id(0)
    if REVERSE:
        reader, writer = program == 0, program == 1
    else:
        reader, writer = program == 1, program == 0
    observed = tl.load(state, reader, other=0)
    previous = tl.atomic_add(state, 1, writer, sem="relaxed", scope="gpu")
    tl.store(out + program, observed + previous)


@triton.jit
def atomic_tickets(state, out, n, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    previous = tl.atomic_add(state + lane * 0, 1, lane < n, sem="acq_rel", scope="gpu")
    tl.store(out + lane, previous, lane < n)


@triton.jit
def atomic_chain(state, out_a, out_b, n, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    first = tl.atomic_add(state + lane, 1, lane < n, sem="relaxed", scope="gpu")
    second = tl.atomic_xor(state + lane, first, lane < n, sem="acq_rel", scope="gpu")
    tl.store(out_a + lane, first, lane < n)
    tl.store(out_b + lane, second, lane < n)
