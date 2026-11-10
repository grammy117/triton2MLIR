"""Real default-library declarations with logical tiling and runtime control."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def external_values(x, y, out_abs, out_sin, out_cos, out_tan, out_asin, out_acos,
                    out_atan, out_sinh, out_cosh, out_tanh, out_asinh, out_acosh,
                    out_atanh, out_log, out_log10, out_log1p, out_exp, out_exp2,
                    out_erf, out_sqrt, out_rsqrt, out_ceil, out_floor, out_trunc,
                    out_atan2, out_pow, n, steps, enabled, B: tl.constexpr):
    indices = tl.program_id(0) * B + tl.arange(0, B)
    mask = indices < n
    a = tl.load(x + indices, mask, other=0)
    b = tl.load(y + indices, mask, other=1)
    for _ in range(steps):
        if enabled:
            a = libdevice.sin(a)
    tl.store(out_abs + indices, libdevice.abs(a), mask)
    tl.store(out_sin + indices, libdevice.sin(a), mask)
    tl.store(out_cos + indices, libdevice.cos(a), mask)
    tl.store(out_tan + indices, libdevice.tan(a), mask)
    tl.store(out_asin + indices, libdevice.asin(a), mask)
    tl.store(out_acos + indices, libdevice.acos(a), mask)
    tl.store(out_atan + indices, libdevice.atan(a), mask)
    tl.store(out_sinh + indices, libdevice.sinh(a), mask)
    tl.store(out_cosh + indices, libdevice.cosh(a), mask)
    tl.store(out_tanh + indices, libdevice.tanh(a), mask)
    tl.store(out_asinh + indices, libdevice.asinh(a), mask)
    tl.store(out_acosh + indices, libdevice.acosh(a), mask)
    tl.store(out_atanh + indices, libdevice.atanh(a), mask)
    tl.store(out_log + indices, libdevice.log(a), mask)
    tl.store(out_log10 + indices, libdevice.log10(a), mask)
    tl.store(out_log1p + indices, libdevice.log1p(a), mask)
    tl.store(out_exp + indices, libdevice.exp(a), mask)
    tl.store(out_exp2 + indices, libdevice.exp2(a), mask)
    tl.store(out_erf + indices, libdevice.erf(a), mask)
    tl.store(out_sqrt + indices, libdevice.sqrt(a), mask)
    tl.store(out_rsqrt + indices, libdevice.rsqrt(a), mask)
    tl.store(out_ceil + indices, libdevice.ceil(a), mask)
    tl.store(out_floor + indices, libdevice.floor(a), mask)
    tl.store(out_trunc + indices, libdevice.trunc(a), mask)
    tl.store(out_atan2 + indices, libdevice.atan2(a, b), mask)
    tl.store(out_pow + indices, libdevice.pow(a, b), mask)
