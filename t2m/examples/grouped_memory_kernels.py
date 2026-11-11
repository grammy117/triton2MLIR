"""Packed logical reads retain cross-slot state and source root selection."""

import triton
import triton.language as tl


@triton.jit
def _read2(p0, p1, l0, l1, n0, n1, c0, c1, it0, it1, WHILE: tl.constexpr):
    first = tl.load(p0, l0 < n0, other=-7)
    second = tl.load(p1, l1 < n1, other=-7)
    a, b = first + second, second - first
    if WHILE:
        step = 0
        while step < it0:
            a += tl.load(p1 + step, (l1 < n1) & (l1 + step < c1), other=-3)
            step += 1
    else:
        for step in range(it0):
            a += tl.load(p1 + step, (l1 < n1) & (l1 + step < c1), other=-3)
    return first, second, a, b


@triton.jit
def _read2_for(p0, p1, l0, l1, n0, n1, c0, c1, it0, it1):
    return _read2(p0, p1, l0, l1, n0, n1, c0, c1, it0, it1, False)


@triton.jit
def _read2_while(p0, p1, l0, l1, n0, n1, c0, c1, it0, it1):
    return _read2(p0, p1, l0, l1, n0, n1, c0, c1, it0, it1, True)


@triton.jit
def grouped_read_state(x, y, out_initial, out_sum, out_final, out_final_sum,
                       n, capacity, iterations, delta, B: tl.constexpr, WHILE: tl.constexpr, SELECT: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    # Each logical program owns its changing state; no global phase is assumed.
    limit = tl.minimum(capacity, (tl.program_id(0) + 1) * B)
    pointer = tl.where(lane % 3 == 0, y + lane, x + lane) if SELECT else x + lane
    if WHILE:
        initial, first = tl.map_elementwise(_read2_while, pointer, lane, n, limit, iterations, pack=2)
    else:
        initial, first = tl.map_elementwise(_read2_for, pointer, lane, n, limit, iterations, pack=2)
    tl.store(x + lane, initial + delta, lane < n)
    if WHILE:
        changed, second = tl.map_elementwise(_read2_while, pointer, lane, n, limit, iterations, pack=2)
    else:
        changed, second = tl.map_elementwise(_read2_for, pointer, lane, n, limit, iterations, pack=2)
    tl.store(out_initial + lane, initial, lane < n)
    tl.store(out_sum + lane, first, lane < n)
    tl.store(out_final + lane, changed, lane < n)
    tl.store(out_final_sum + lane, second, lane < n)


@triton.jit
def _swap2(p0, p1, m0, m1):
    return p1, p0, m1, m0


@triton.jit
def grouped_pointer_result(x, y, out, n, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    pointer = tl.where(lane % 3 == 0, y + lane, x + lane)
    selected, active = tl.map_elementwise(_swap2, pointer, lane < n, pack=2)
    value = tl.load(selected, active, other=-9)
    tl.store(out + lane, value, lane < n)
