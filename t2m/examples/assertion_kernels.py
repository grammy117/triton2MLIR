"""Runtime source assertions inside the original tile recurrence/branch."""

import triton
import triton.language as tl


@triton.jit
def checked_copy(x, out, n, steps, limit, B: tl.constexpr, MODE: tl.constexpr, RANK: tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    for step in range(steps):
        if step % 2 == 0 and (MODE != 3 or limit >= 0):
            if MODE != 1:
                tl.device_assert(n >= 0, "source count must be nonnegative")
            active = lane < n
            value = tl.load(x + lane, active, other=0)
            if MODE != 0:
                condition = (~active) | (value <= limit)
                if RANK == 2:
                    condition = tl.reshape(condition, (2, B // 2))
                tl.device_assert(condition, "source tile exceeds limit")
            tl.store(out + lane, value + step, active)
