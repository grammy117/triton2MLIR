"""Program-local colliding updates followed by synchronized observations."""

import triton
import triton.language as tl


@triton.jit
def atomic_phases(x, state, old, seen, n, steps, B: tl.constexpr, CROSS: tl.constexpr = False):
    local = tl.arange(0, B)
    program = tl.program_id(0)
    lane = program * B + local
    slot = local % (B // 2)
    if not CROSS:
        slot += program * (B // 2)
    active = lane < n
    value = tl.load(x + lane, active, other=0)
    initial = tl.full((B,), -7, state.dtype.element_ty)
    tl.store(old + lane, initial, active)
    tl.store(seen + lane, initial, active)
    for _ in range(steps):
        previous = tl.atomic_add(state + slot, value, active, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        after = tl.load(state + slot, active, other=-3)
        tl.store(old + lane, previous, active)
        tl.store(seen + lane, after, active)
        tl.debug_barrier()
