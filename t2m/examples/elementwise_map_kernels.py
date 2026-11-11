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
