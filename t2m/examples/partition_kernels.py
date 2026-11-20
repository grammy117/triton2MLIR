"""Conditional and masked alternatives preserve ordered private writes."""

import triton
import triton.language as tl


@triton.jit
def partition_transform(x, bias, out, seen, n, control, MODE: tl.constexpr):
    reused = tl.load(bias)
    q = control != 0
    for i in range(n):
        value = tl.load(x + i) ^ reused
        p = value > 0
        if MODE == "branch":
            if p:
                tl.store(out + i, value ^ 1)
            else:
                tl.store(out + i, value ^ 2)
        elif MODE == "nested-branch":
            if p:
                if q:
                    tl.store(out + i, value ^ 1)
                else:
                    tl.store(out + i, value ^ 2)
            else:
                tl.store(out + i, value ^ 4)
        elif MODE == "nested" or MODE == "nested-missing":
            tl.store(out + i, value ^ 1, p & q)
            tl.store(out + i, value ^ 2, p & ~q)
            if MODE != "nested-missing":
                tl.store(out + i, value ^ 4, ~p)
        elif MODE == "select":
            selected = tl.where(q, p, ~p)
            tl.store(out + i, value ^ 1, selected)
            tl.store(out + i, value ^ 2, ~selected)
        elif MODE == "overlap":
            tl.store(out + i, value ^ 1, p | q)
            tl.store(out + i, value ^ 2, ~p | ~q)
        elif MODE == "missing":
            if p:
                tl.store(out + i, value ^ 1)
        else:
            tl.store(out + i, value ^ 1, p)
            if MODE == "observed":
                earlier = tl.load(out + i, p, other=-17)
            elif MODE == "early":
                earlier = tl.load(out + i)
            if MODE == "duplicate":
                tl.store(out + i, value ^ 2, p)
            elif MODE == "independent":
                tl.store(out + i, value ^ 2, ~q)
            elif MODE == "different-index":
                tl.store(out + i + 1, value ^ 2, ~p & (i + 1 < n))
            else:
                tl.store(out + i, value ^ 2, ~p)
        if MODE == "observed" or MODE == "early":
            observation = earlier
        else:
            observation = tl.load(out + i)
        tl.store(seen + i, observation)


@triton.jit
def zero_trip_write(out, n):
    for _i in range(n):
        tl.store(out, 1)
    tl.store(out + 1, tl.load(out))


@triton.jit
def partition_rows(x, bias, out, seen, rows, cols, control, ORDER: tl.constexpr):
    outer_count = rows if ORDER == 0 else cols
    inner_count = cols if ORDER == 0 else rows
    for outer in range(outer_count):
        reused = tl.load(bias + outer)
        for inner in range(inner_count):
            offset = outer * cols + inner if ORDER == 0 else inner * cols + outer
            value = tl.load(x + offset) ^ reused
            p = value > 0
            if control != 0:
                tl.store(out + offset, value ^ 1, p)
                tl.store(out + offset, value ^ 2, ~p)
            else:
                if p:
                    tl.store(out + offset, value ^ 8)
                else:
                    tl.store(out + offset, value ^ 4)
        start = outer * cols if ORDER == 0 else outer
        tl.store(seen + outer, tl.load(out + start, inner_count > 0, other=-17))


@triton.jit
def float_partition(x, out, seen, n):
    for i in range(n):
        positive = tl.load(x + i) > 0.0
        tl.store(out + i, 11, positive)
        tl.store(out + i, -23, ~positive)
        tl.store(seen + i, tl.load(out + i))


@triton.jit
def ordered_float_gap(x, out, seen, n):
    for i in range(n):
        value = tl.load(x + i)
        tl.store(out + i, 11, value > 0.0)
        tl.store(out + i, -23, value <= 0.0)
        tl.store(seen + i, tl.load(out + i))
