"""Original integer-address loops, branches and observable allocation state."""

import triton
import triton.language as tl


@triton.jit
def _address_step(address, total, state, lane, n, dtype: tl.constexpr):
    pointer = address.to(tl.pointer_type(dtype))
    value = tl.load(pointer, lane < n, other=-17)
    total = total ^ value
    tl.store(state + lane, total, lane < n)
    return total


@triton.jit
def integer_address_state(x, y, state, distance, n, capacity, steps, origin, stride, choice,
                          B: tl.constexpr, WIDTH: tl.constexpr, MODE: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    offset = (origin.to(tl.int64) + lane.to(tl.int64)) * (WIDTH // 8)
    if MODE == "branch":
        if choice > 0:
            address = x.to(tl.uint64) + offset
        else:
            address = y.to(tl.uint64) + offset
    else:
        address = x.to(tl.uint64) + offset
    initial = address
    total = tl.load(state + lane, lane < n, other=0)
    if MODE == "for" or MODE == "branch":
        for _ in range(steps):
            total = _address_step(address, total, state, lane, n, x.dtype.element_ty)
            address = address + stride.to(tl.int64) * (WIDTH // 8)
    else:
        step = 0
        while step < steps:
            total = _address_step(address, total, state, lane, n, x.dtype.element_ty)
            offset = offset + stride.to(tl.int64) * (WIDTH // 8)
            if MODE == "alternating":
                if (step + choice) % 2 == 0:
                    address = x.to(tl.uint64) + offset
                else:
                    address = y.to(tl.uint64) + offset
            else:
                address = address + stride.to(tl.int64) * (WIDTH // 8)
            step += 1
    if MODE == "alternating":
        # This source exposes relative coordinates, not the chosen root's
        # absolute integer. The actual alternating address is used by loads.
        delta = offset - (origin.to(tl.int64) + lane.to(tl.int64)) * (WIDTH // 8)
    else:
        delta = (address - initial).to(tl.int64)
    tl.store(distance + lane, delta, lane < n)
