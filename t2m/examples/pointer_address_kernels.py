"""Pointer integers used only to form logical byte-displaced addresses."""

import triton
import triton.language as tl


@triton.jit
def displaced_words(
    x, y, out, n, steps, origin, stride, capacity, B: tl.constexpr, WIDTH: tl.constexpr
):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    element: tl.constexpr = (
        tl.int64
        if WIDTH == 64
        else tl.int32
        if WIDTH == 32
        else tl.int16
        if WIDTH == 16
        else tl.int8
    )
    word_bytes: tl.constexpr = WIDTH // 8
    for row in range(steps):
        index = (
            origin.to(tl.int64)
            + row.to(tl.int64) * stride.to(tl.int64)
            + lane.to(tl.int64)
        )
        displacement = index * word_bytes
        # Select the absolute integer, not the original pointer. Both bases
        # and offsets have to survive re-raising independently.
        first = x.to(tl.uint64) + displacement
        second = y.to(tl.uint64) + displacement
        address = tl.where(lane % 2 == 0, first, second)
        pointer = address.to(tl.pointer_type(element))
        value = tl.load(pointer, (lane < n) & (lane % 3 != 1), other=-17)
        output_address = out.to(tl.uint64) + displacement
        output_pointer = output_address.to(tl.pointer_type(element))
        tl.store(output_pointer, value ^ (row + 1).to(element), lane < n)


@triton.jit
def scalar_displacement(x, out, offset):
    address = x.to(tl.uint64) - offset.to(tl.int64) * 4
    pointer = address.to(tl.pointer_type(tl.int32))
    tl.store(out, tl.load(pointer))


@triton.jit
def masked_scalar_memory(x, out, offset, destination, active):
    value = tl.load(x + offset, active, other=-17)
    tl.store(out + destination, value, active)
