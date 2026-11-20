"""Observe completed rows, columns and planes before whole-output coverage."""

import triton
import triton.language as tl


@triton.jit
def progressive_transform(x, bias, out, first, last, rows, cols,
                          ORDER: tl.constexpr, MODE: tl.constexpr):
    outer_count = rows if ORDER == 0 else cols
    inner_count = cols if ORDER == 0 else rows
    for outer in range(outer_count):
        reused = tl.load(bias + outer)
        for inner in range(inner_count):
            row = outer if ORDER == 0 else inner
            col = inner if ORDER == 0 else outer
            offset = row * cols + col
            if MODE == "before-read":
                reused ^= tl.load(out + offset)
            value = tl.load(x + offset) ^ reused
            if MODE == "masked":
                tl.store(out + offset, value, inner > 0)
            else:
                tl.store(out + offset, value)
            if MODE == "point":
                tl.store(out + offset, tl.load(out + offset) ^ 1)
        origin = outer * cols if ORDER == 0 else outer
        end = outer * cols + cols - 1 if ORDER == 0 else (rows - 1) * cols + outer
        if MODE == "future":
            origin += cols if ORDER == 0 else 1
        if MODE == "unmasked":
            observed = tl.load(out + origin)
        elif MODE == "or-mask":
            observed = tl.load(out + origin, (inner_count > 0) | (outer_count > 0), other=-19)
        elif MODE == "and-mask":
            observed = tl.load(out + origin, (inner_count > 0) & (outer % 2 == 0), other=-19)
        else:
            observed = tl.load(out + origin, inner_count > 0, other=-19)
        if MODE == "reread":
            observed = tl.full((), 0, tl.int64)
            for inner in range(inner_count):
                offset = outer * cols + inner if ORDER == 0 else inner * cols + outer
                observed ^= tl.load(out + offset)
        final = tl.load(out + end, inner_count > 0, other=-23)
        tl.store(first + outer, observed)
        tl.store(last + outer, final)


@triton.jit
def progressive_volume(x, bias, out, first, last, a, b, c, ORDER: tl.constexpr):
    d0 = a if ORDER == 0 else c
    d1 = b if ORDER == 0 else a
    d2 = c if ORDER == 0 else b
    for outer in range(d0):
        reused = tl.load(bias + outer)
        for middle in range(d1):
            for inner in range(d2):
                row = outer if ORDER == 0 else middle
                col = middle if ORDER == 0 else inner
                channel = inner if ORDER == 0 else outer
                offset = (row * b + col) * c + channel
                tl.store(out + offset, tl.load(x + offset) ^ reused)
        start = outer * b * c if ORDER == 0 else outer
        end = (outer + 1) * b * c - 1 if ORDER == 0 else (a * b - 1) * c + outer
        active = d1 * d2 > 0
        tl.store(first + outer, tl.load(out + start, active, other=-19))
        tl.store(last + outer, tl.load(out + end, active, other=-23))
