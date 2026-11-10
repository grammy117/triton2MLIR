"""Indexed integer atomics with original old values and ordinary observations."""

import triton
import triton.language as tl


@triton.jit
def indexed_cas(state, comparison, value, old, B: tl.constexpr, SCALAR: tl.constexpr,
                SEM: tl.constexpr, SCOPE: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    expected = tl.load(comparison + lane)
    desired = tl.load(value + lane)
    previous = tl.atomic_cas(state + lane, expected, desired, sem=SEM, scope=SCOPE)
    tl.store(old + lane, previous)


@triton.jit
def selected_cas(a, b, comparison, value, old, choice,
                 B: tl.constexpr, SCALAR: tl.constexpr, LANE: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    pointer = a + lane
    if LANE:
        pointer = tl.where(lane % 2 == 0, pointer, b + lane)
    else:
        if choice:
            pointer = b + lane
    expected = tl.load(comparison + lane)
    desired = tl.load(value + lane)
    previous = tl.atomic_cas(pointer, expected, desired, sem="acq_rel", scope="gpu")
    tl.store(old + lane, previous)


@triton.jit
def matrix_cas(state, comparison, value, old):
    row = tl.program_id(0) * 2 + tl.arange(0, 2)
    column = tl.arange(0, 4)
    lane = row[:, None] * 4 + column[None, :]
    previous = tl.atomic_cas(state + lane, tl.load(comparison + lane), tl.load(value + lane), sem="acquire", scope="cta")
    tl.store(old + lane, previous)


@triton.jit
def region_cas(state, out, n, steps, enabled, shift,
               B: tl.constexpr, SCALAR: tl.constexpr, WHILE: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    pointer = state + lane + shift
    total = (lane * 0).to(state.dtype.element_ty)
    if enabled:
        if WHILE:
            i = 0
            while i < steps:
                comparison = tl.full((B,) if not SCALAR else (), 0, state.dtype.element_ty) + (i - 11).to(state.dtype.element_ty)
                value = tl.full((B,) if not SCALAR else (), 0, state.dtype.element_ty) + (i + 1).to(state.dtype.element_ty)
                previous = tl.atomic_cas(pointer, comparison, value, sem="relaxed", scope="gpu")
                total += previous
                pointer += n
                i += 1
        else:
            for i in range(steps):
                comparison = tl.full((B,) if not SCALAR else (), 0, state.dtype.element_ty) + (i - 11).to(state.dtype.element_ty)
                value = tl.full((B,) if not SCALAR else (), 0, state.dtype.element_ty) + (i + 1).to(state.dtype.element_ty)
                previous = tl.atomic_cas(pointer, comparison, value, sem="relaxed", scope="gpu")
                total += previous
                pointer += n
        last = tl.atomic_cas(pointer, total, total + 7, sem="release", scope="sys")
        total += last
    tl.store(out + lane, total, lane < n)


@triton.jit
def cas_collision(state, old):
    program = tl.program_id(0)
    previous = tl.atomic_cas(state, tl.full((), 0, state.dtype.element_ty), tl.full((), 1, state.dtype.element_ty), sem="acq_rel", scope="gpu")
    tl.store(old + program, previous)


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


@triton.jit
def region_atomic(a, b, out, n, steps, enabled, choice, shift,
                  B: tl.constexpr, SCALAR: tl.constexpr, WHILE: tl.constexpr,
                  SELECT: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    pointer = a + lane + shift
    if SELECT == "lane":
        pointer = tl.where(lane % 2 == 0, pointer, b + lane + shift)
    elif SELECT == "uniform":
        if choice:
            pointer = b + lane + shift
    total = (lane * 0).to(a.dtype.element_ty)
    if enabled:
        if WHILE:
            i = 0
            while i < steps:
                if i % 2 == 0:
                    previous = tl.atomic_add(pointer, i + 1, lane < n, sem="relaxed", scope="gpu")
                else:
                    previous = tl.atomic_xor(pointer, i + 1, lane < n, sem="acquire", scope="cta")
                total += previous
                pointer += n
                i += 1
        else:
            for i in range(steps):
                if i % 2 == 0:
                    previous = tl.atomic_add(pointer, i + 1, lane < n, sem="relaxed", scope="gpu")
                else:
                    previous = tl.atomic_xor(pointer, i + 1, lane < n, sem="acquire", scope="cta")
                total += previous
                pointer += n
        last = tl.atomic_xor(pointer, total, lane < n, sem="release", scope="sys")
        total += last
    tl.store(out + lane, total, lane < n)


@triton.jit
def indexed_word_atomic(x, index, state, old, n, m, B: tl.constexpr, OP: tl.constexpr,
                        WIDTH: tl.constexpr, SCALAR: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    signed: tl.constexpr = tl.int32 if WIDTH == 32 else tl.int64
    unsigned: tl.constexpr = tl.uint32 if WIDTH == 32 else tl.uint64
    word: tl.constexpr = unsigned if OP == "umax" or OP == "umin" else signed
    at = tl.load(index + lane, lane < n, other=-123)
    value = tl.load(x + lane, lane < n, other=0).to(word, bitcast=True)
    pointer = (state + at).to(tl.pointer_type(word), bitcast=True)
    active = (lane < n) & (lane % 3 != 1)
    if OP == "add":
        previous = tl.atomic_add(pointer, value, active, sem="acq_rel", scope="gpu")
    elif OP == "and":
        previous = tl.atomic_and(pointer, value, active, sem="acq_rel", scope="gpu")
    elif OP == "or":
        previous = tl.atomic_or(pointer, value, active, sem="acq_rel", scope="gpu")
    elif OP == "xor":
        previous = tl.atomic_xor(pointer, value, active, sem="acq_rel", scope="gpu")
    elif OP == "max" or OP == "umax":
        previous = tl.atomic_max(pointer, value, active, sem="acq_rel", scope="gpu")
    elif OP == "min" or OP == "umin":
        previous = tl.atomic_min(pointer, value, active, sem="acq_rel", scope="gpu")
    else:
        previous = tl.atomic_xchg(pointer, value, active, sem="acq_rel", scope="gpu")
    tl.store(old + lane, previous.to(old.dtype.element_ty, bitcast=True), lane < n)


@triton.jit
def constant_word_atomic(state, old, B: tl.constexpr, WIDTH: tl.constexpr, OP: tl.constexpr, VALUE: tl.constexpr):
    lane = tl.arange(0, B)
    word: tl.constexpr = tl.int32 if WIDTH == 32 else tl.int64
    pointer = (state + lane).to(tl.pointer_type(word), bitcast=True)
    value = tl.full((B,), VALUE, word)
    if OP == "and":
        previous = tl.atomic_and(pointer, value)
    elif OP == "max":
        previous = tl.atomic_max(pointer, value)
    elif OP == "min":
        previous = tl.atomic_min(pointer, value)
    elif OP == "exch":
        previous = tl.atomic_xchg(pointer, value)
    else:
        previous = tl.atomic_add(pointer, value)
    tl.store(old + lane, previous.to(old.dtype.element_ty, bitcast=True))


@triton.jit
def region_word_atomic(a, b, out, n, steps, enabled, choice, shift,
                       B: tl.constexpr, WIDTH: tl.constexpr, SCALAR: tl.constexpr,
                       WHILE: tl.constexpr, SELECT: tl.constexpr):
    word: tl.constexpr = tl.int32 if WIDTH == 32 else tl.int64
    region_atomic(a.to(tl.pointer_type(word), bitcast=True),
                  b.to(tl.pointer_type(word), bitcast=True),
                  out.to(tl.pointer_type(word), bitcast=True),
                  n, steps, enabled, choice, shift, B=B, SCALAR=SCALAR, WHILE=WHILE, SELECT=SELECT)


@triton.jit
def repeated_region_word_atomic(a, b, out, n, steps, enabled, choice, shift,
                                B: tl.constexpr, WIDTH: tl.constexpr, SCALAR: tl.constexpr,
                                WHILE: tl.constexpr, SELECT: tl.constexpr):
    word: tl.constexpr = tl.int32 if WIDTH == 32 else tl.int64
    first = a.to(tl.pointer_type(word), bitcast=True)
    second = b.to(tl.pointer_type(word), bitcast=True)
    output = out.to(tl.pointer_type(word), bitcast=True)
    region_atomic(first, second, output, n, steps, enabled, choice, shift,
                  B=B, SCALAR=SCALAR, WHILE=WHILE, SELECT=SELECT)
    region_atomic(first, second, output + n, n, steps, enabled, ~choice, shift,
                  B=B, SCALAR=SCALAR, WHILE=WHILE, SELECT=SELECT)


@triton.jit
def region_fadd(a, b, out, n, steps, enabled, choice, shift,
                B: tl.constexpr, SCALAR: tl.constexpr, WHILE: tl.constexpr,
                SELECT: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    pointer = a + lane + shift
    if SELECT == "lane":
        pointer = tl.where(lane % 2 == 0, pointer, b + lane + shift)
    elif SELECT == "uniform":
        if choice:
            pointer = b + lane + shift
    total = (lane * 0).to(a.dtype.element_ty)
    if enabled:
        if WHILE:
            i = 0
            while i < steps:
                previous = tl.atomic_add(pointer, i + 1, lane < n, sem="acquire", scope="cta")
                total += previous
                pointer += n
                i += 1
        else:
            for i in range(steps):
                previous = tl.atomic_add(pointer, i + 1, lane < n, sem="relaxed", scope="gpu")
                total += previous
                pointer += n
        last = tl.atomic_add(pointer, total, lane < n, sem="release", scope="sys")
        total += last
    tl.store(out + lane, total, lane < n)


@triton.jit
def constant_fadd32(state, old, B: tl.constexpr, WORD: tl.constexpr):
    lane = tl.arange(0, B)
    value = tl.full((B,), WORD, tl.uint32).to(tl.float32, bitcast=True)
    previous = tl.atomic_add(state + lane, value, sem="acq_rel", scope="gpu")
    tl.store(old + lane, previous)


@triton.jit
def constant_fadd(state, old, B: tl.constexpr, WIDTH: tl.constexpr, WORD: tl.constexpr):
    lane = tl.arange(0, B)
    integer: tl.constexpr = tl.uint32 if WIDTH == 32 else tl.uint64
    value = tl.full((B,), WORD, integer).to(state.dtype.element_ty, bitcast=True)
    previous = tl.atomic_add(state + lane, value, sem="acq_rel", scope="gpu")
    tl.store(old + lane, previous)


@triton.jit
def atomic_word_update(state, old, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    previous = tl.atomic_add(state + lane, (lane + 1).to(state.dtype.element_ty), sem="acq_rel", scope="gpu")
    tl.store(old + lane, previous)


@triton.jit
def mixed_storage_atomic(integer_state, floating_state, old_integer, old_word,
                         B: tl.constexpr, WIDTH: tl.constexpr):
    word: tl.constexpr = tl.int32 if WIDTH == 32 else tl.int64
    atomic_word_update(integer_state, old_integer, B=B)
    atomic_word_update(floating_state.to(tl.pointer_type(word), bitcast=True),
                       old_word.to(tl.pointer_type(word), bitcast=True), B=B)


@triton.jit
def word_cas(state, comparison, value, old, B: tl.constexpr, WIDTH: tl.constexpr, SCALAR: tl.constexpr):
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    indexed_cas(state.to(tl.pointer_type(word), bitcast=True), comparison, value,
                old.to(tl.pointer_type(word), bitcast=True), B=B, SCALAR=SCALAR, SEM="acq_rel", SCOPE="gpu")


@triton.jit
def selected_word_cas(a, b, comparison, value, old, choice,
                      B: tl.constexpr, WIDTH: tl.constexpr, SCALAR: tl.constexpr, LANE: tl.constexpr):
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    selected_cas(a.to(tl.pointer_type(word), bitcast=True), b.to(tl.pointer_type(word), bitcast=True),
                 comparison, value, old.to(tl.pointer_type(word), bitcast=True), choice,
                 B=B, SCALAR=SCALAR, LANE=LANE)


@triton.jit
def matrix_word_cas(state, comparison, value, old, WIDTH: tl.constexpr):
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    matrix_cas(state.to(tl.pointer_type(word), bitcast=True), comparison, value,
               old.to(tl.pointer_type(word), bitcast=True))


@triton.jit
def constant_word_cas(state, old, B: tl.constexpr, WIDTH: tl.constexpr, COMPARISON: tl.constexpr, VALUE: tl.constexpr):
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    lane = tl.arange(0, B)
    pointer = state.to(tl.pointer_type(word), bitcast=True) + lane
    comparison = tl.full((B,), COMPARISON, word)
    value = tl.full((B,), VALUE, word)
    previous = tl.atomic_cas(pointer, comparison, value, sem="acq_rel", scope="gpu")
    tl.store(old.to(tl.pointer_type(word), bitcast=True) + lane, previous)


@triton.jit
def region_word_cas(state, out, n, steps, enabled, shift,
                    B: tl.constexpr, WIDTH: tl.constexpr, SCALAR: tl.constexpr, WHILE: tl.constexpr):
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    region_cas(state.to(tl.pointer_type(word), bitcast=True), out.to(tl.pointer_type(word), bitcast=True),
               n, steps, enabled, shift, B=B, SCALAR=SCALAR, WHILE=WHILE)


@triton.jit
def mixed_storage_cas(integer_state, floating_state, comparison, value, old_integer, old_word,
                       B: tl.constexpr, WIDTH: tl.constexpr):
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    indexed_cas(integer_state, comparison, value, old_integer, B=B, SCALAR=False, SEM="acq_rel", SCOPE="gpu")
    indexed_cas(floating_state.to(tl.pointer_type(word), bitcast=True), comparison, value,
                old_word.to(tl.pointer_type(word), bitcast=True), B=B, SCALAR=False, SEM="acq_rel", SCOPE="gpu")
