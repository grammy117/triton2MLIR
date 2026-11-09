"""Finite buffer selection with original memory-event ordering."""

import triton
import triton.language as tl


@triton.jit
def selected_access(x, y, out, n, m, choice, B: tl.constexpr, SCALAR: tl.constexpr,
                    LANES: tl.constexpr, STORE_SELECTED: tl.constexpr):
    if SCALAR:
        lane = tl.program_id(0)
    else:
        lane = tl.program_id(0) * B + tl.arange(0, B)
    if LANES:
        pointer = tl.where((lane + choice) % 2 == 0, x + lane, y + m - 1 - lane)
    else:
        if choice == 0:
            pointer = x + lane
        else:
            pointer = y + m - 1 - lane
    # This write must remain before the selected observation.
    original = tl.load(x + lane, lane < n, other=0)
    tl.store(x + lane, original + 3, lane < n)
    if STORE_SELECTED:
        value = tl.load(pointer, lane < n, other=7)
        tl.store(pointer, value * 2, lane < n)
    value = tl.load(pointer, lane < n, other=7)
    tl.store(out + lane, value, lane < n)


@triton.jit
def _next_pointer(x, y, lane, m, step, choice, LANES: tl.constexpr, SWITCH: tl.constexpr):
    if SWITCH:
        pointer = y + m - 1 - lane
    elif LANES:
        pointer = tl.where((lane + step + choice) % 2 == 0, x + lane, y + m - 1 - lane)
    else:
        if (step + choice) % 2 == 0:
            pointer = y + m - 1 - lane
        else:
            pointer = x + lane
    return pointer


@triton.jit
def selected_loop(x, y, out, n, m, steps, choice, B: tl.constexpr, FOR_LOOP: tl.constexpr,
                  LANES: tl.constexpr, SWITCH: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    pointer = x + lane
    acc = tl.full((B,), 0, tl.float32)
    if FOR_LOOP:
        for step in range(steps):
            acc += tl.load(pointer, lane < n, other=0)
            pointer = _next_pointer(x, y, lane, m, step, choice, LANES, SWITCH)
    else:
        step = 0
        while step < steps:
            acc += tl.load(pointer, lane < n, other=0)
            pointer = _next_pointer(x, y, lane, m, step, choice, LANES, SWITCH)
            step += 1
    acc += tl.load(pointer, lane < n, other=0)
    tl.store(out + lane, acc, lane < n)


@triton.jit
def selected_shape(x, y, out, n, m, B: tl.constexpr):
    block = tl.program_id(0) * (2 * B)
    lane = block + tl.arange(0, B)
    pointers = tl.reshape(tl.trans(tl.join(x + lane, y + m - 1 - lane)), (2 * B,))
    index = block + tl.arange(0, 2 * B)
    value = tl.load(pointers, index < n, other=-7)
    tl.store(out + index, value, index < n)
