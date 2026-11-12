"""Real early returns retain per-program mutation and tiled recurrences."""

import triton
import triton.language as tl


@triton.jit
def early_return_words(x, state, n, steps, mode, B: tl.constexpr):
    pid = tl.program_id(0)
    if (pid + mode) % 2 == 0:
        return
    lane = pid * B + tl.arange(0, B)
    value = tl.load(state + lane, lane < n, other=0)
    data = tl.load(x + lane, lane < n, other=0)
    for i in range(steps):
        value = value ^ (data + (i + 1).to(data.dtype))
    tl.store(state + lane, value, lane < n)
