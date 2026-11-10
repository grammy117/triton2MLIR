"""Logical reinterpretation with pointer recurrences and finite root selection."""

import triton
import triton.language as tl


@triton.jit
def word_view_memory(x, y, out, n, steps, B: tl.constexpr, WIDTH: tl.constexpr,
                     SELECT: tl.constexpr, BEFORE: tl.constexpr, WHILE: tl.constexpr, SCALAR: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    p, q = x + lane, out + lane
    if BEFORE:
        p = p.to(tl.pointer_type(word), bitcast=True)
    if SELECT:
        alternate = y + lane
        if BEFORE:
            alternate = alternate.to(tl.pointer_type(word), bitcast=True)
        p = tl.where(lane % 2 == 0, p, alternate)
    if not BEFORE:
        p = p.to(tl.pointer_type(word), bitcast=True)
    q = q.to(tl.pointer_type(word), bitcast=True)
    sign = tl.full((), -(1 << (WIDTH - 1)), word)
    if WHILE:
        i = 0
        while i < steps:
            value = tl.load(p, (lane < n) & (lane % 3 != 1), other=-17)
            tl.store(q, value ^ sign, lane < n)
            p += n
            q += n
            i += 1
    else:
        for _ in range(steps):
            value = tl.load(p, (lane < n) & (lane % 3 != 1), other=-17)
            tl.store(q, value ^ sign, lane < n)
            p += n
            q += n


@triton.jit
def mixed_word_event(p, before, after, lane, n, WIDTH: tl.constexpr):
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    active = (lane < n) & (lane % 3 != 1)
    fill = (lane * 7 - 17).to(word)
    sign = tl.full((), -(1 << (WIDTH - 1)), word)
    previous = tl.load(p, active, other=fill)
    tl.store(before + lane, previous, lane < n)
    tl.store(p, previous ^ sign, active)
    observed = tl.load(p, active, other=fill)
    tl.store(after + lane, observed, lane < n)


@triton.jit
def mixed_storage_updates(x, y, before, after, final, n, steps, choice, enabled, shift,
                           B: tl.constexpr, WIDTH: tl.constexpr, SCALAR: tl.constexpr,
                           WHILE: tl.constexpr, LANE: tl.constexpr):
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    a = x.to(tl.pointer_type(word), bitcast=True) + lane
    b = y.to(tl.pointer_type(word), bitcast=True) + lane + shift
    select = (lane % 2 == 0) == choice if LANE else choice
    p, q = tl.where(select, a, b), tl.where(select, b, a)
    first = before.to(tl.pointer_type(word), bitcast=True)
    second = after.to(tl.pointer_type(word), bitcast=True)
    if enabled:
        if WHILE:
            i = 0
            while i < steps:
                mixed_word_event(p, first + i * n, second + i * n, lane, n, WIDTH=WIDTH)
                p, q = q + n, p + n
                i += 1
        else:
            for i in range(steps):
                mixed_word_event(p, first + i * n, second + i * n, lane, n, WIDTH=WIDTH)
                p, q = q + n, p + n
    observed = tl.load(p, (lane < n) & enabled & (steps < 4), other=-17)
    tl.store(final.to(tl.pointer_type(word), bitcast=True) + lane, observed, lane < n)


@triton.jit
def three_storage_permutation(x, y, z, out, n, steps,
                               B: tl.constexpr, WIDTH: tl.constexpr, WHILE: tl.constexpr, REVERSE: tl.constexpr):
    word: tl.constexpr = tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    lane = tl.program_id(0) * B + tl.arange(0, B)
    a, b, c = x + lane, y.to(tl.pointer_type(word), bitcast=True) + lane, z + lane
    p, q, r = tl.where(lane % 2 == 0, a, b), tl.where(lane % 3 == 0, b, c), tl.where(lane % 5 == 0, c, a)
    if WHILE:
        i = 0
        while i < steps:
            tl.store(out + i * n + lane, tl.load(p, lane < n, other=-17), lane < n)
            if REVERSE:
                p, q, r = r + n, p + n, q + n
            else:
                p, q, r = q + n, r + n, p + n
            i += 1
    else:
        for i in range(steps):
            tl.store(out + i * n + lane, tl.load(p, lane < n, other=-17), lane < n)
            if REVERSE:
                p, q, r = r + n, p + n, q + n
            else:
                p, q, r = q + n, r + n, p + n
    tl.store(out + steps * n + lane, tl.load(p, lane < n, other=-17), lane < n)
