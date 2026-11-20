"""Atomic old values and actual volatile reads over invocation-private state."""

import triton
import triton.language as tl


@triton.jit
def atomic_volatile_updates(x, state, old, seen, n, steps, B: tl.constexpr,
                           SYNC: tl.constexpr = True, CROSS: tl.constexpr = False):
    origin = tl.program_id(0) * B
    local = tl.arange(0, B)
    lane = origin + local
    active = lane < n
    value = tl.load(x + lane, active, other=0, volatile=True)
    _unused = tl.load(x + lane, active, other=0, volatile=True)
    slot = origin // 2 + local % (B // 2) if SYNC else lane
    initial = tl.full((B,), -7, old.dtype.element_ty)
    tl.store(old + lane, initial, active)
    tl.store(seen + lane, initial, active)
    for _ in range(steps):
        before = tl.atomic_add(state + slot, value, active, sem="acq_rel", scope="gpu")
        if SYNC:
            tl.debug_barrier()
        read_slot = slot + (B // 2 if SYNC else B) if CROSS else slot
        after = tl.load(state + read_slot, active, other=-3, volatile=True)
        tl.store(old + lane, before, active)
        tl.store(seen + lane, after, active)
        if SYNC:
            tl.debug_barrier()


@triton.jit
def atomic_volatile_cas(x, state, old, seen, n, B: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    active = lane < n
    value = tl.load(x + lane, active, other=0, volatile=True)
    # CAS has no source mask. Inactive lanes still update valid padded state.
    before = tl.atomic_cas(state + lane, value, value + 1, sem="acq_rel", scope="gpu")
    after = tl.load(state + lane, active, other=-3, volatile=True)
    tl.store(old + lane, before, active)
    tl.store(seen + lane, after, active)
