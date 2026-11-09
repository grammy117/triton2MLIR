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
