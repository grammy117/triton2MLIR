"""High precision mathematical references; never an input to compilation.

mpmath evaluates transcendental values at 110 decimal digits. Exact rational
rounding supplies f32/f64 words; source recurrence rounds at each iteration.
Exceptional values and signed-zero branches are explicit. This is a logical
standard-math reference, not an implementation of NVIDIA's approximations.
"""

from fractions import Fraction
import math

import mpmath as mp
import numpy as np

from .scalar_math_reference import rounded

NAMES = ("abs", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
         "asinh", "acosh", "atanh", "log", "log10", "log1p", "exp", "exp2", "erf",
         "sqrt", "rsqrt", "ceil", "floor", "trunc", "atan2", "pow")
EXACT = {"abs", "ceil", "floor", "trunc"}
SIGNED_ZERO = {"sin", "tan", "asin", "atan", "sinh", "tanh", "asinh", "atanh", "log1p", "erf", "sqrt", "ceil", "floor", "trunc"}


def inputs(dtype, n=37):
    dt = np.float32 if dtype == "f32" else np.float64
    tiny = np.nextafter(dt(0), dt(1), dtype=dt)
    values = np.asarray([0., -0., 1., -1., .5, -.5, 2., -2., tiny, -tiny,
                         np.inf, -np.inf, np.nan, .125, -.125, 3., -3.,
                         1.5, -1.5, 16., -16., 100., -100., 710., -750.,
                         np.finfo(dt).tiny, -np.finfo(dt).tiny], dt)
    exponents = np.asarray([0., -0., 1., -1., 2., -2., 3., -3., .5, -.5,
                            np.inf, -np.inf, np.nan], dt)
    return np.resize(values, n), np.resize(exponents, n)


def evaluate(name, x, y):
    negative = math.copysign(1., x) < 0
    if name == "abs":
        return abs(x)
    if name == "pow":
        if y == 0 or x == 1 or x == -1 and math.isinf(y):
            return 1.
        if math.isnan(x) or math.isnan(y):
            return mp.nan
        if math.isinf(y):
            return mp.inf if (abs(x) > 1) == (y > 0) else 0.
        odd = math.isfinite(y) and y.is_integer() and bool(int(y) & 1)
        if x == 0:
            return math.copysign(0. if y > 0 else math.inf, -1. if negative and odd else 1.)
        if math.isinf(x):
            return math.copysign(math.inf if y > 0 else 0., -1. if negative and odd else 1.)
        if x < 0 and math.isfinite(y) and not y.is_integer():
            return mp.nan
        return mp.power(mp.mpf(x), mp.mpf(y))
    if math.isnan(x) or name == "atan2" and math.isnan(y):
        return mp.nan
    if name == "atan2":
        if x == 0:
            return mp.sign(-1 if negative else 1) * mp.pi if math.copysign(1., y) < 0 else x
        if math.isinf(x) and math.isinf(y):
            return (-1 if negative else 1) * mp.pi * (mp.mpf(3)/4 if y < 0 else mp.mpf(1)/4)
        if math.isinf(y):
            return (-1 if negative else 1)*mp.pi if y < 0 else math.copysign(0., x)
        return mp.atan2(mp.mpf(x), mp.mpf(y))
    if x == 0 and name in SIGNED_ZERO:
        return x
    if name in ("log", "log10") and x == 0 or name == "log1p" and x == -1:
        return -mp.inf
    if name == "rsqrt" and x == 0:
        return math.copysign(math.inf, x)
    if name == "atanh" and abs(x) == 1:
        return math.copysign(math.inf, x)
    if name in ("exp", "exp2", "sinh", "cosh") and abs(x) > 10000:
        return (-mp.inf if x < 0 else mp.inf) if name == "sinh" else (mp.inf if name == "cosh" or x > 0 else mp.mpf(0))
    a = mp.mpf(x)
    if name == "rsqrt":
        return 1/mp.sqrt(a)
    if name == "exp2":
        return mp.power(2, a)
    if name in ("ceil", "floor", "trunc"):
        value = mp.ceil(a) if name == "ceil" else mp.floor(a) if name == "floor" else (mp.ceil(a) if x < 0 else mp.floor(a))
        return -0. if not value and negative else value
    return getattr(mp, name)(a)


def scalar(name, x, y, dtype):
    dt, integer = (np.float32, np.uint32) if dtype == "f32" else (np.float64, np.uint64)
    with mp.workdps(110):
        try:
            result = evaluate(name, float(x), float(y))
        except (ValueError, ZeroDivisionError):
            result = mp.nan
        if isinstance(result, mp.mpc):
            result = result.real if not result.imag else mp.nan
        if mp.isnan(result):
            return dt(np.nan)
        if mp.isinf(result):
            return dt(-np.inf if result < 0 else np.inf)
        if not result:
            return dt(-0. if isinstance(result, float) and math.copysign(1., result) < 0 else 0.)
        result = mp.mpf(result)
        # Clamp before constructing a rational; huge exponential exponents
        # must never become huge Python integer shifts in a reference tool.
        if abs(result) > mp.mpf(float(np.finfo(dt).max)) * 2:
            return dt(-np.inf if result < 0 else np.inf)
        if abs(result) < mp.power(2, -151 if dtype == "f32" else -1076):
            return dt(-0. if result < 0 else 0.)
        sign, mantissa, exponent, _ = result._mpf_
        rational = Fraction((-1 if sign else 1)*mantissa) * Fraction(2)**exponent
        word = rounded(rational, dtype)
        return np.asarray(word, integer).view(dt)[()]


def reference(x, y, dtype, steps=0, enabled=True):
    current = x.copy()
    if enabled:
        for _ in range(max(steps, 0)):
            current = np.asarray([scalar("sin", value, 0., dtype) for value in current], current.dtype)
    return [np.asarray([scalar(name, a, b, dtype) for a,b in zip(current,y,strict=True)], current.dtype) for name in NAMES]


def policy(dtype):
    return {"schema":"t2m.effect-reference.v2", "termination":"return", "stdout":"", "stderr":"",
            "results":{"archive":"reference.npz", "comparisons":[
                {"kind":"ieee"} if name in EXACT else {"kind":"values","rtol":4e-6 if dtype == "f32" else 4e-14,"atol":0.}
                for name in NAMES]}}
