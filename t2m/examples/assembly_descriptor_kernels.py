"""Logical descriptor state with explicitly selected source word computation."""

import triton
import triton.language as tl


@triton.jit
def assembly_descriptor_state(
    first, second, m, n, steps, capacity, ASM: tl.constexpr, WORDS64: tl.constexpr
):
    row = tl.program_id(0) * 2
    col = tl.program_id(1) * 16
    for _ in range(steps):
        value = first.load([row, col])
        value = tl.inline_asm_elementwise(
            ASM,
            constraints="=l,l" if WORDS64 else "=r,r",
            args=[value],
            dtype=tl.int64 if WORDS64 else tl.int32,
            is_pure=True,
            pack=1,
        )
        second.store([row, col], value)


@triton.jit
def assembly_descriptor_words(
    first,
    second,
    m,
    n,
    steps,
    capacity,
    shift,
    ASM: tl.constexpr,
    WORDS64: tl.constexpr,
):
    row = tl.program_id(0) * 2
    col = tl.program_id(1) * 16
    for _ in range(steps):
        value = first.load([row, col])
        a, b = tl.inline_asm_elementwise(
            ASM,
            constraints="=l,=l,l,l,r" if WORDS64 else "=r,=r,r,r,r",
            args=[value, value ^ 7, shift],
            dtype=(tl.int64, tl.int64) if WORDS64 else (tl.int32, tl.int32),
            is_pure=True,
            pack=1,
        )
        second.store([row, col], a - 3 * b)
