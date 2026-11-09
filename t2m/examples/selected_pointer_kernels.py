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


@triton.jit
def selected_fills(x, y, fills, out, n, m, choice, B: tl.constexpr, OMIT_OTHER: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    pointer = tl.where((lane + choice) % 2 == 0, x + lane, y + m - 1 - lane)
    active = (lane < n) & (lane % 3 != 1)
    if OMIT_OTHER:
        value = tl.load(pointer, active)
        tl.store(out + lane, value, active)
    else:
        fill = tl.load(fills + lane, lane < n, other=0)
        value = tl.load(pointer, active, other=fill)
        tl.store(out + lane, value, lane < n)


@triton.jit
def selected_words(x, y, encoded, out, n, m, choice, B: tl.constexpr, CANCEL: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    key = tl.load(encoded + lane, lane < n, other=0)
    if CANCEL:
        pointer = tl.where((lane + choice) % 2 == 0, x + lane, y + m - 1 - lane)
        pointer += key
        pointer += key
        pointer -= 2 * key.to(tl.int64)
    else:
        index = key * 4 + 1
        pointer = tl.where((lane + choice) % 2 == 0, x + index, y + m - 1 - index)
    value = tl.load(pointer, lane < n, other=0)
    tl.store(out + lane, value, lane < n)


@triton.jit
def selected_permute(x, y, out, n, m, choice, B: tl.constexpr, MODE: tl.constexpr):
    if MODE == 1:
        lane = tl.program_id(0) * B + tl.arange(0, B)
        choose = (lane + choice) % 2 == 0
        pointers = tl.join(tl.where(choose, x + lane, y + m - 1 - lane),
                           tl.where(choose, y + m - 1 - lane, x + lane))
        left, right = tl.split(pointers)
        value = tl.load(left, lane < n, other=0) - 2 * tl.load(right, lane < n, other=0)
        tl.store(out + lane, value, lane < n)
    else:
        block = tl.program_id(0) * (2 * B)
        lane = block + tl.arange(0, B)
        pointers = tl.cat(x + lane, y + m - 1 - lane, can_reorder=True)
        index = block + tl.arange(0, 2 * B)
        if MODE == 2:
            selected = (tl.arange(0, 2 * B) * 6 + choice) % (2 * B)
            pointers = tl.gather(pointers, selected, 0)
        value = tl.load(pointers, index < n, other=-7)
        tl.store(out + index, value, index < n)


@triton.jit
def selected_unsplat(x, y, out, n, choice, MODE: tl.constexpr):
    index = tl.program_id(0)
    lane = tl.full((1,), index, tl.int32)
    if MODE == 0:
        pointer = x + lane
    elif MODE == 1:
        if choice == 0:
            pointer = x + lane
        else:
            pointer = y + n - 1 - lane
    else:
        pointer = tl.where((lane + choice) % 2 == 0, x + lane, y + n - 1 - lane)
    scalar_pointer = tl.reshape(pointer, ())
    value = tl.load(scalar_pointer, index < n, other=-7)
    tl.store(out + index, value, index < n)
