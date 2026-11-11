"""Distinct typed source arguments deliberately share one owning allocation."""

import triton
import triton.language as tl


@triton.jit
def descriptor_word_updates(first, second, m, n, steps, capacity):
    row = tl.program_id(0) * 2
    col = tl.program_id(1) * 16
    for _ in range(steps):
        loaded = first.load([row, col])
        second.store([row, col], loaded.to(tl.int32, bitcast=True) ^ 1)
        forwarded = first.load([row, col])
        second.store([row, col], forwarded.to(tl.int32, bitcast=True) ^ 2)


@triton.jit
def allocation_word_updates(
    floating,
    integer,
    n,
    steps,
    capacity,
    B: tl.constexpr,
    WIDTH: tl.constexpr,
    WHILE: tl.constexpr,
):
    lanes = tl.program_id(0) * B + tl.arange(0, B)
    active = lanes < n
    if WHILE:
        iteration = 0
        while iteration < steps:
            loaded = tl.load(floating + lanes, active, other=0.0)
            words = loaded.to(
                tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64,
                bitcast=True,
            )
            tl.store(integer + lanes, words ^ 1, active)
            # A second read must observe the preceding differently typed store.
            forwarded = tl.load(floating + lanes, active, other=0.0)
            restored = forwarded.to(
                tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64,
                bitcast=True,
            )
            tl.store(integer + lanes, restored ^ 2, active)
            iteration += 1
    else:
        for _ in range(steps):
            loaded = tl.load(floating + lanes, active, other=0.0)
            words = loaded.to(
                tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64,
                bitcast=True,
            )
            tl.store(integer + lanes, words ^ 1, active)
            forwarded = tl.load(floating + lanes, active, other=0.0)
            restored = forwarded.to(
                tl.int16 if WIDTH == 16 else tl.int32 if WIDTH == 32 else tl.int64,
                bitcast=True,
            )
            tl.store(integer + lanes, restored ^ 2, active)
