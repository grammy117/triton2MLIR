"""Two external logical views may observe and update one allocation root."""

import triton
import triton.language as tl


@triton.jit
def descriptor_state_updates(first, second, m, n, steps, capacity):
    row = tl.program_id(0) * 2
    col = tl.program_id(1) * 16
    for _ in range(steps):
        value = first.load([row, col])
        second.store([row, col], value + 1)


@triton.jit
def unused_descriptor(unused, state, n, capacity):
    lanes = tl.arange(0, 16)
    value = tl.load(state + lanes, lanes < n, other=0)
    tl.store(state + lanes, value + 1, lanes < n)
