"""Kernels whose observable result is a diagnostic or normal return."""

import triton
import triton.language as tl


@triton.jit
def check_scalar(value, steps):
    for i in range(steps):
        if i % 2 == 0:
            tl.device_assert(value >= 0, "effect scalar must be nonnegative")


@triton.jit
def check_tensor(x, n, limit, B:tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0,B)
    active = lane < n
    value = tl.load(x + lane,active,other=0)
    tl.device_assert((~active) | (value <= limit), "effect tile exceeds limit")


@triton.jit
def constant_check(FAIL:tl.constexpr):
    tl.device_assert(not FAIL, "constant source assertion")


@triton.jit
def fail_before_store(out,B:tl.constexpr):
    tl.device_assert(False,"failure precedes real output")
    tl.store(out + tl.arange(0,B),0)


@triton.jit
def check_diagnostic(value, steps, MESSAGE:tl.constexpr):
    for i in range(steps):
        if i % 2 == 0:
            tl.device_assert(value >= 0, MESSAGE)


@triton.jit
def check_tensor_diagnostic(x, n, limit, MESSAGE:tl.constexpr, B:tl.constexpr):
    lane = tl.program_id(0) * B + tl.arange(0,B)
    active = lane < n
    value = tl.load(x + lane,active,other=0)
    condition = ((~active) | (value <= limit)).reshape((2,B // 2))
    tl.device_assert(condition, MESSAGE)


@triton.jit
def copy_diagnostic(x, out, allow, MESSAGE:tl.constexpr, B:tl.constexpr):
    lane = tl.arange(0,B)
    tl.store(out + lane,tl.load(x + lane))
    tl.device_assert(allow, MESSAGE)


@triton.jit
def atomic_diagnostic(out, allow, MESSAGE:tl.constexpr):
    tl.atomic_cas(out,1,3,sem="acq_rel",scope="gpu")
    tl.device_assert(allow, MESSAGE)
