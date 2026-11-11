"""Mapped scalar branches, tuple results and loop recurrences."""

import triton
import triton.language as tl


@triton.jit
def _map_initial(a, b):
    if b != 0:
        value = a / b if a.dtype.is_floating() else a // b
    else:
        value = tl.full((), -7, a.dtype)
    return value


@triton.jit
def _map_scalar(a, b, iterations):
    value = _map_initial(a, b)
    for step in range(iterations):
        value = value + b * step
    return value, a + b


@triton.jit
def _map_scalar_while(a, b, iterations):
    value = _map_initial(a, b)
    step = 0
    while step < iterations:
        value = value + b * step
        step += 1
    return value, a + b


@triton.jit
def elementwise_map_values(x, y, out_a, out_b, n, iterations, B: tl.constexpr, WHILE_LOOP: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    a, b = tl.load(x + lane, lane < n, other=0), tl.load(y + lane, lane < n, other=0)
    if WHILE_LOOP:
        first, second = tl.map_elementwise(_map_scalar_while, a, b, iterations)
    else:
        first, second = tl.map_elementwise(_map_scalar, a, b, iterations)
    tl.store(out_a + lane, first, lane < n)
    tl.store(out_b + lane, second, lane < n)


@triton.jit
def _packed_pair(a0, a1, b0, b1):
    return a0 + b0, a1 - b1


@triton.jit
def packed_map_values(x, y, out):
    lane = tl.arange(0, 8)
    a, b = tl.load(x + lane), tl.load(y + lane)
    value, = tl.map_elementwise(_packed_pair, a, b, pack=2)
    tl.store(out + lane, value)


@triton.jit
def _packed_value(a, b, step):
    divisor = tl.where(b != 0, b, 1)
    quotient = a / divisor if a.dtype.is_floating() else a // divisor
    return tl.where(b != 0, quotient, -7) + step, a + b


@triton.jit
def _packed_independent2(a0, a1, b0, b1, s0, s1):
    q0, r0 = _packed_value(a0, b0, s0)
    q1, r1 = _packed_value(a1, b1, s1)
    return q0, q1, r0, r1


@triton.jit
def _packed_independent4(a0, a1, a2, a3, b0, b1, b2, b3, s0, s1, s2, s3):
    q0, r0 = _packed_value(a0, b0, s0)
    q1, r1 = _packed_value(a1, b1, s1)
    q2, r2 = _packed_value(a2, b2, s2)
    q3, r3 = _packed_value(a3, b3, s3)
    return q0, q1, q2, q3, r0, r1, r2, r3


@triton.jit
def independent_packed_maps(x, y, out_a, out_b, n, iterations, B: tl.constexpr, PACK: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if PACK == 4:
        lane = tl.reshape(lane, (4, B // 4))
    a = tl.load(x + lane, lane < n, other=0)
    b = tl.load(y + lane, lane < n, other=0)
    for step in range(iterations):
        if PACK == 2:
            a, b = tl.map_elementwise(_packed_independent2, a, b, step, pack=PACK)
        else:
            a, b = tl.map_elementwise(_packed_independent4, a, b, step, pack=PACK)
    tl.store(out_a + lane, a, lane < n)
    tl.store(out_b + lane, b, lane < n)


@triton.jit
def _packed_cross_slot(a0, a1):
    return a0 + a1, a1 + a0


@triton.jit
def coupled_packed_map(x, out):
    lane = tl.arange(0, 8)
    value, = tl.map_elementwise(_packed_cross_slot, tl.load(x + lane), pack=2)
    tl.store(out + lane, value)


@triton.jit
def _map_pointer_load(pointer):
    return tl.load(pointer)


@triton.jit
def pointer_load_map(x, out):
    lane = tl.arange(0, 8)
    value = tl.map_elementwise(_map_pointer_load, x + lane)
    tl.store(out + lane, value)


@triton.jit
def _mapped_read(pointer, lane, n, capacity, iterations, WHILE: tl.constexpr):
    if lane < n:
        initial = tl.load(pointer)
        value = initial
        if WHILE:
            step = 0
            while step < iterations:
                value += tl.load(pointer + step, lane + step < capacity, other=-3)
                step += 1
        else:
            for step in range(iterations):
                value += tl.load(pointer + step, lane + step < capacity, other=-3)
    else:
        initial = tl.full((), -7, pointer.dtype.element_ty)
        value = initial
    return initial, value


@triton.jit
def _mapped_read_for(pointer, lane, n, capacity, iterations):
    return _mapped_read(pointer, lane, n, capacity, iterations, False)


@triton.jit
def _mapped_read_while(pointer, lane, n, capacity, iterations):
    return _mapped_read(pointer, lane, n, capacity, iterations, True)


@triton.jit
def memory_map_state(x, out_initial, out_sum, out_final, out_final_sum, n, capacity, iterations, delta,
                     B: tl.constexpr, WHILE: tl.constexpr):
    lane = tl.arange(0, B)
    if WHILE:
        initial, first = tl.map_elementwise(_mapped_read_while, x + lane, lane, n, capacity, iterations)
    else:
        initial, first = tl.map_elementwise(_mapped_read_for, x + lane, lane, n, capacity, iterations)
    tl.store(x + lane, initial + delta, lane < n)
    if WHILE:
        changed, second = tl.map_elementwise(_mapped_read_while, x + lane, lane, n, capacity, iterations)
    else:
        changed, second = tl.map_elementwise(_mapped_read_for, x + lane, lane, n, capacity, iterations)
    # Both original snapshots are consumed after the intervening mutation.
    tl.store(out_initial + lane, initial, lane < n)
    tl.store(out_sum + lane, first, lane < n)
    tl.store(out_final + lane, changed, lane < n)
    tl.store(out_final_sum + lane, second, lane < n)


@triton.jit
def _map_masked_read(pointer, active):
    return tl.load(pointer, active, other=-7)


@triton.jit
def _map_pick_read(first, second, choice, active):
    pointer = tl.where(choice, first, second)
    return tl.load(pointer, active, other=-7)


@triton.jit
def _map_pick_pointer(first, second, choice):
    return tl.where(choice, first, second)


@triton.jit
def _map_pick_mixed(first, second, choice, active):
    pointer = tl.where(choice, first, second)
    return pointer, tl.load(pointer, active, other=-7)


@triton.jit
def selected_map_reads(x, y, out, observed, n, capacity_x, capacity_y, choice, B: tl.constexpr, STAGE: tl.constexpr):
    lane = tl.arange(0, B)
    choose = (lane % 2 == 0) == choice
    if STAGE == 0:
        pointer = tl.where(choose, x + lane, y + lane)
        value = tl.map_elementwise(_map_masked_read, pointer, lane < n)
        old = value
    elif STAGE == 1:
        value = tl.map_elementwise(_map_pick_read, x + lane, y + lane, choose, lane < n)
        old = value
    elif STAGE == 2:
        pointer = tl.map_elementwise(_map_pick_pointer, x + lane, y + lane, choose)
        value = tl.load(pointer, lane < n, other=-7)
        old = value
    else:
        pointer, old = tl.map_elementwise(_map_pick_mixed, x + lane, y + lane, choose, lane < n)
        value = tl.load(pointer, lane < n, other=-7)
    tl.store(out + lane, value, lane < n)
    tl.store(observed + lane, old, lane < n)
