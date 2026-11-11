"""Real Triton descriptor parameters; no device descriptor allocation or launch."""

import triton
import triton.language as tl


@triton.jit
def external_descriptor_transfer(
    desc, values, indices, count, y, capacity, SCATTER: tl.constexpr
):
    slot = tl.program_id(0) * 8 + tl.arange(0, 8)
    selected = tl.load(indices + slot, slot < count, other=-1)
    column = tl.arange(0, 16)
    if SCATTER:
        tile = tl.load(
            values + slot[:, None] * 16 + column[None, :],
            slot[:, None] < count,
            other=0,
        )
        desc.scatter(tile, selected, y)
    else:
        tile = desc.gather(selected, y)
        tl.store(
            values + slot[:, None] * 16 + column[None, :], tile, slot[:, None] < count
        )


@triton.jit
def external_descriptor_reduce(
    desc, values, tiles, steps, row, y, capacity, KIND: tl.constexpr
):
    slot = (
        tl.program_id(0) * 32
        + tl.arange(0, 2)[:, None] * 16
        + tl.arange(0, 16)[None, :]
    )
    tile = tl.load(values + slot)
    for _ in range(steps):
        if KIND == "add":
            desc.atomic_add([row, y], tile)
        elif KIND == "min":
            desc.atomic_min([row, y], tile)
        elif KIND == "max":
            desc.atomic_max([row, y], tile)
        elif KIND == "and":
            desc.atomic_and([row, y], tile)
        elif KIND == "or":
            desc.atomic_or([row, y], tile)
        else:
            desc.atomic_xor([row, y], tile)


@triton.jit
def external_descriptor_block(
    desc,
    values,
    first,
    second,
    third,
    capacity,
    SHAPE: tl.constexpr,
    ELEMENTS: tl.constexpr,
    STORE: tl.constexpr,
):
    if len(SHAPE) == 1:
        offsets = [first]
    elif len(SHAPE) == 2:
        offsets = [first, second]
    else:
        offsets = [first, second, third]
    slot = tl.arange(0, ELEMENTS)
    if STORE:
        tile = tl.reshape(tl.load(values + slot), SHAPE)
        desc.store(offsets, tile)
    else:
        tile = desc.load(offsets)
        tl.store(values + slot, tl.reshape(tile, (ELEMENTS,)))


@triton.jit
def external_descriptor_matmul(x, y, initial, out, m, n, k, a_count, b_count):
    row = tl.program_id(0) * 16
    col = tl.program_id(1) * 16
    r = row + tl.arange(0, 16)
    c = col + tl.arange(0, 16)
    mask = (r[:, None] < m) & (c[None, :] < n)
    accumulator = tl.load(initial + r[:, None] * n + c[None, :], mask, other=0)
    for step in range(tl.cdiv(k, 32)):
        left = x.load([row, step * 32])
        right = y.load([step * 32, col])
        accumulator = tl.dot(left, right, accumulator, input_precision="ieee")
    tl.store(out + r[:, None] * n + c[None, :], accumulator, mask)
