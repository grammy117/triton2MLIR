"""Logical indexed descriptor transfers, independent of physical TMA mapping."""

import triton
import triton.language as tl


@triton.jit
def indexed_descriptor_copy(
    x,
    indices,
    out,
    rows,
    cols,
    count,
    y,
    stride,
    B: tl.constexpr,
    C: tl.constexpr,
    SCATTER: tl.constexpr,
    PADDING: tl.constexpr = "zero",
):
    slot = tl.program_id(0) * B + tl.arange(0, B)
    selected = tl.load(indices + slot, slot < count, other=-1)
    column = tl.arange(0, C)
    if SCATTER:
        value = tl.load(
            x + slot[:, None] * C + column[None, :], slot[:, None] < count, other=0
        )
        desc = tl.make_tensor_descriptor(out, [rows, cols], [stride, 1], [1, C])
        desc.scatter(value, selected, y)
    else:
        desc = tl.make_tensor_descriptor(
            x, [rows, cols], [stride, 1], [1, C], padding_option=PADDING
        )
        value = desc.gather(selected, y)
        tl.store(
            out + slot[:, None] * C + column[None, :], value, slot[:, None] < count
        )


@triton.jit
def descriptor_updates(
    x, out, rows, cols, tiles, steps, row, y, stride, KIND: tl.constexpr
):
    slot = (
        tl.program_id(0) * 32
        + tl.arange(0, 2)[:, None] * 16
        + tl.arange(0, 16)[None, :]
    )
    value = tl.load(x + slot)
    desc = tl.make_tensor_descriptor(out, [rows, cols], [stride, 1], [2, 16])
    for _ in range(steps):
        if KIND == "add":
            desc.atomic_add([row, y], value)
        elif KIND == "min":
            desc.atomic_min([row, y], value)
        elif KIND == "max":
            desc.atomic_max([row, y], value)
        elif KIND == "and":
            desc.atomic_and([row, y], value)
        elif KIND == "or":
            desc.atomic_or([row, y], value)
        else:
            desc.atomic_xor([row, y], value)
