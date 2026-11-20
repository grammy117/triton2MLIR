"""Private volatile observations across complete cross-lane memory phases."""

import triton
import triton.language as tl


@triton.jit
def volatile_exchange(state, out, n, steps, B: tl.constexpr, CROSS: tl.constexpr = False):
    origin = tl.program_id(0) * B
    local = tl.arange(0, B)
    row, column = origin + local[:, None], local[None, :]
    pointer = row * B + column
    # A real observation even when the loop has no iterations or its result is unused.
    _unused = tl.load(state + pointer, row < n, other=0, volatile=True)
    tl.store(out + pointer, tl.full((B, B), -7, out.dtype.element_ty), row < n)
    for step in range(steps):
        before = tl.load(state + pointer, row < n, other=0, volatile=True)
        tl.store(state + pointer, before + step + 1, row < n)
        tl.debug_barrier()
        transposed_row = origin + local[None, :]
        if CROSS:
            transposed_row += B
        after = tl.load(state + transposed_row * B + local[:, None],
                        transposed_row < n, other=-3, volatile=True)
        tl.store(out + pointer, after, row < n)
        tl.debug_barrier()
