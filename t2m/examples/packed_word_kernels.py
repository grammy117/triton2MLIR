"""Packed source words with ordinary tensor loops, masks and loaded reuse."""

import triton
import triton.language as tl


@triton.jit
def packed_word_state(x, y, out_a, out_b, n, steps,
                      ASM: tl.constexpr, CONSTRAINTS: tl.constexpr,
                      PACK: tl.constexpr, B: tl.constexpr, RANK2: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if RANK2:
        lane = tl.reshape(lane, (2, B // 2))
    a = tl.load(x + lane, lane < n, other=0)
    b = tl.load(y + lane, lane < n, other=0)
    for _ in range(steps):
        a, b = tl.inline_asm_elementwise(ASM, constraints=CONSTRAINTS, args=[a, b],
            dtype=(a.dtype, b.dtype), is_pure=True, pack=PACK)
    tl.store(out_a + lane, a, lane < n)
    tl.store(out_b + lane, b, lane < n)


@triton.jit
def packed_word_cast(x, out, n, ASM: tl.constexpr, CONSTRAINTS: tl.constexpr,
                     PACK: tl.constexpr, OUT_WIDTH: tl.constexpr, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + lane, lane < n, other=0)
    if OUT_WIDTH == 8:
        result = tl.inline_asm_elementwise(ASM, constraints=CONSTRAINTS, args=[value],
            dtype=tl.int8, is_pure=True, pack=PACK)
    elif OUT_WIDTH == 16:
        result = tl.inline_asm_elementwise(ASM, constraints=CONSTRAINTS, args=[value],
            dtype=tl.int16, is_pure=True, pack=PACK)
    else:
        result = tl.inline_asm_elementwise(ASM, constraints=CONSTRAINTS, args=[value],
            dtype=tl.int32, is_pure=True, pack=PACK)
    tl.store(out + lane, result, lane < n)
