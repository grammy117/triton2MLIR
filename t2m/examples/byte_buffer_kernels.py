"""Typed views of a byte cache retain source addptr units and word events."""

import triton
import triton.language as tl


@triton.jit
def byte_buffer_selected(
    a, b, out, n, steps, capacity, origin, stride, advance, choice, B: tl.constexpr
):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    pa = (a + origin + lane * stride).to(tl.pointer_type(tl.int32), bitcast=True)
    pb = (b + origin + lane * stride).to(tl.pointer_type(tl.int32), bitcast=True)
    iteration = 0
    while iteration < steps:
        selected = tl.where((lane % 2 == 0) == choice, pa, pb)
        current = tl.load(selected, lane < n, other=0)
        updated = current ^ 3
        tl.store(selected, updated, lane < n)
        tl.store(out + iteration * n + lane, updated, lane < n)
        pa += advance
        pb += advance
        choice = ~choice
        iteration += 1


@triton.jit
def byte_buffer_fills(
    raw,
    out,
    n,
    padded,
    capacity,
    origin,
    stride,
    fill,
    B: tl.constexpr,
    TYPE: tl.constexpr,
    WIDTH: tl.constexpr,
):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    target: tl.constexpr = (
        tl.float16
        if TYPE == "f16"
        else tl.bfloat16
        if TYPE == "bf16"
        else tl.float32
        if TYPE == "f32"
        else tl.float64
    )
    integer: tl.constexpr = (
        tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    )
    pointer = (raw + origin + lane * stride).to(tl.pointer_type(target), bitcast=True)
    value = tl.load(pointer, lane < n, other=fill.to(target, bitcast=True))
    tl.store(out + lane, value.to(integer, bitcast=True))


@triton.jit
def byte_buffer_updates(
    raw,
    out,
    n,
    steps,
    capacity,
    origin,
    stride,
    advance,
    B: tl.constexpr,
    TYPE: tl.constexpr,
    WIDTH: tl.constexpr,
    SCALAR: tl.constexpr,
    NARROW: tl.constexpr,
):
    lane = tl.program_id(0) if SCALAR else tl.program_id(0) * B + tl.arange(0, B)
    target: tl.constexpr = (
        tl.int16
        if TYPE == "i16"
        else tl.int32
        if TYPE == "i32"
        else tl.int64
        if TYPE == "i64"
        else tl.float16
        if TYPE == "f16"
        else tl.bfloat16
        if TYPE == "bf16"
        else tl.float32
        if TYPE == "f32"
        else tl.float64
    )
    integer: tl.constexpr = (
        tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64
    )
    pointer = (raw + origin + lane * stride).to(tl.pointer_type(target), bitcast=True)
    for iteration in range(steps):
        value = tl.load(pointer, lane < n, other=0)
        word = (
            value
            if TYPE == "i16" or TYPE == "i32" or TYPE == "i64"
            else value.to(integer, bitcast=True)
        )
        changed = word ^ 3
        stored = (
            changed
            if TYPE == "i16" or TYPE == "i32" or TYPE == "i64"
            else changed.to(target, bitcast=True)
        )
        tl.store(pointer, stored, lane < n)
        if NARROW:
            byte_pointer = pointer.to(tl.pointer_type(tl.int8), bitcast=True)
            tl.store(byte_pointer + 1, 0x5A, lane < n)
        current = tl.load(pointer, lane < n, other=0)
        result = (
            current
            if TYPE == "i16" or TYPE == "i32" or TYPE == "i64"
            else current.to(integer, bitcast=True)
        )
        tl.store(out + iteration * n + lane, result, lane < n)
        pointer += advance
