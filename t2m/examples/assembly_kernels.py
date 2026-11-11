"""Source assembly is an explicit input; the frontend chooses no target ISA."""

import triton
import triton.language as tl


@triton.jit
def assembly_word_state(x, y, shifts, out_a, out_b, n, iterations,
                        ASM: tl.constexpr, B: tl.constexpr, WORDS64: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if WORDS64:
        lane = tl.reshape(lane, (4, B // 4))
    a = tl.load(x + lane, lane < n, other=0)
    b = tl.load(y + lane, lane < n, other=0)
    count = tl.load(shifts + lane, lane < n, other=0)
    for _ in range(iterations):
        if WORDS64:
            a, b = tl.inline_asm_elementwise(ASM, constraints="=l,=l,l,l,r", args=[a, b, count],
                                           dtype=(tl.int64, tl.int64), is_pure=True, pack=1)
        else:
            a, b = tl.inline_asm_elementwise(ASM, constraints="=r,=r,r,r,r", args=[a, b, count],
                                           dtype=(tl.int32, tl.int32), is_pure=True, pack=1)
    tl.store(out_a + lane, a, lane < n)
    tl.store(out_b + lane, b, lane < n)
