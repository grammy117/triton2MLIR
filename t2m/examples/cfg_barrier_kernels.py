"""Uniform early exits preserve synchronized tile updates and observations."""

import triton
import triton.language as tl


@triton.jit
def early_tile_exchange(state, old, seen, n, steps, mode, B: tl.constexpr,
                        ATOMIC: tl.constexpr = False, VOLATILE: tl.constexpr = False):
    pid = tl.program_id(0)
    if (pid + mode) % 2 == 0:
        return
    local = tl.arange(0, B)
    row = pid * B + local
    pointer = row[:, None] * B + local[None, :]
    transpose_row = pid * B + local[None, :]
    transpose_pointer = transpose_row * B + local[:, None]
    for step in range(steps):
        increment = (step + 1).to(state.dtype.element_ty)
        if ATOMIC:
            before = tl.atomic_add(state + pointer, increment, row[:, None] < n,
                                   sem="acq_rel", scope="gpu")
        else:
            before = tl.load(state + pointer, row[:, None] < n, other=0, volatile=VOLATILE)
            tl.store(state + pointer, before + increment, row[:, None] < n)
        tl.debug_barrier()
        after = tl.load(state + transpose_pointer, transpose_row < n, other=-3, volatile=VOLATILE)
        tl.store(old + pointer, before, row[:, None] < n)
        tl.store(seen + pointer, after, row[:, None] < n)
        tl.debug_barrier()
