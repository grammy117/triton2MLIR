"""Original rectangular loop traversals with reused outer-axis values."""

import triton
import triton.language as tl


@triton.jit
def nested_overwrite(x, bias, out, seen, rows, cols, count, enabled,
                     ORDER: tl.constexpr, MODE: tl.constexpr):
    outer_count = rows if ORDER == 0 else cols
    inner_count = cols if ORDER == 0 else rows
    if MODE == "different":
        outer_count = count
    for outer in range(0, outer_count, 2 if MODE == "step" else 1):
        reused = tl.load(bias + outer)
        limit = outer + 1 if MODE == "dependent" else inner_count
        for inner in range(limit):
            row = outer if ORDER == 0 else inner
            col = inner if ORDER == 0 else outer
            offset = row * cols if MODE == "duplicate" else row * cols + col
            value = tl.load(x + offset) ^ reused
            if MODE == "before-read":
                value ^= tl.load(out + offset)
            if MODE == "branch":
                if enabled:
                    tl.store(out + offset, value)
            elif MODE == "masked":
                tl.store(out + offset, value, inner + 1 < limit)
            else:
                tl.store(out + offset, value)
    active = rows * cols > 0
    first = tl.load(out, active, other=-19)
    last = tl.load(out + rows * cols - 1, active, other=-23)
    tl.store(seen, first)
    tl.store(seen + 1, last)


@triton.jit
def nested_volume_overwrite(x, bias, out, seen, a, b, c, ORDER: tl.constexpr):
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
    active = a * b * c > 0
    first = tl.load(out, active, other=-19)
    last = tl.load(out + a * b * c - 1, active, other=-23)
    tl.store(seen, first)
    tl.store(seen + 1, last)


@triton.jit
def branch_overwrite(x, bias, out, seen, rows, cols, count, enabled,
                     ORDER: tl.constexpr, MODE: tl.constexpr):
    if enabled:
        for i in range(rows * cols):
            value = tl.load(x + i) ^ tl.load(bias + i // cols)
            tl.store(out + i, value)
    else:
        nested_overwrite(x, bias, out, seen, rows, cols, count, enabled, 0, "full")
    active = rows * cols > 0
    tl.store(seen, tl.load(out, active, other=-19))
    tl.store(seen + 1, tl.load(out + rows * cols - 1, active, other=-23))


@triton.jit
def rectangular_transform(x, bias, out, seen, rows, cols, ORDER: tl.constexpr):
    nested_overwrite(x, bias, out, seen, rows, cols, rows, 1, ORDER, "full")


@triton.jit
def empty_result(out, seen, enabled):
    value = tl.load(out, enabled != 0, other=7)
    tl.store(seen, value)
