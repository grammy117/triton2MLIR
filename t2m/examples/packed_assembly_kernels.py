"""Real packed source programs retain loops, loaded reuse and logical tiles."""

import triton
import triton.language as tl


@triton.jit
def packed_assembly_state(x, y, out_a, out_b, n, steps,
                          ASM: tl.constexpr, CONSTRAINTS: tl.constexpr,
                          PACK: tl.constexpr, B: tl.constexpr,
                          WORDS64: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    if WORDS64:
        lane = tl.reshape(lane, (2, B // 2))
    a = tl.load(x + lane, lane < n, other=0)
    b = tl.load(y + lane, lane < n, other=0)
    for _ in range(steps):
        a, b = tl.inline_asm_elementwise(
            ASM, constraints=CONSTRAINTS, args=[a, b],
            dtype=(a.dtype, b.dtype), is_pure=True, pack=PACK)
    tl.store(out_a + lane, a, lane < n)
    tl.store(out_b + lane, b, lane < n)
