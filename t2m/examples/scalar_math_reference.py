"""Independent exact-rational IEEE reference; never a compiler input.

Division uses integer quotient/remainder rounding. Square root compares exact
integer squares with rounding midpoints. No floating division, square root or
compiler-emitted implementation supplies the expected result.
"""

from fractions import Fraction
from math import isqrt

import numpy as np

from t2m.tensor_storage import DTYPES,FLOAT_FORMATS


def fields(dtype):
    exponent,fraction = FLOAT_FORMATS[dtype]
    return exponent,fraction,(1 << (exponent-1))-1,((1 << exponent)-1) << fraction,1 << (exponent+fraction)


def decode(word,dtype):
    exponent,fraction,bias,mask,sign = fields(dtype)
    word = int(word)
    mantissa = word & ((1 << fraction)-1)
    exp = (word & mask) >> fraction
    if exp == (1 << exponent)-1:
        return "nan" if mantissa else "inf",None
    value = Fraction((1 << fraction)+mantissa if exp else mantissa)
    power = (exp-bias if exp else 1-bias)-fraction
    value *= Fraction(2)**power
    return "finite",-value if word & sign else value


def log2_floor(value):
    exponent = value.numerator.bit_length()-value.denominator.bit_length()
    return exponent-(value < Fraction(2)**exponent)


def pack(significand,exponent,negative,dtype):
    _,fraction,bias,mask,sign = fields(dtype)
    if significand >= 1 << (fraction+1):
        significand >>= 1
        exponent += 1
    if exponent > bias:
        return mask | (sign if negative else 0)
    exp = exponent+bias if significand >= 1 << fraction else 0
    return (sign if negative else 0) | exp << fraction | significand & ((1 << fraction)-1)


def rounded(value,dtype,*,negative_zero=False):
    _,fraction,bias,_,sign = fields(dtype)
    if not value:
        return sign if negative_zero else 0
    negative = value < 0
    value = abs(value)
    exponent = max(log2_floor(value),1-bias)
    scaled = value / Fraction(2)**(exponent-fraction)
    quotient,remainder = divmod(scaled.numerator,scaled.denominator)
    quotient += 2*remainder > scaled.denominator or 2*remainder == scaled.denominator and quotient % 2
    return pack(quotient,exponent,negative,dtype)


def divide(left,right,dtype):
    _,fraction,_,inf,sign = fields(dtype)
    lk,lv = decode(left,dtype);rk,rv = decode(right,dtype)
    nan = inf | 1 << (fraction-1)
    negative = bool((int(left)^int(right)) & sign)
    if lk == "nan" or rk == "nan" or lk == rk == "inf" or lk == rk == "finite" and not lv and not rv:
        return nan
    if lk == "inf" or rk == "finite" and not rv:
        return inf | (sign if negative else 0)
    if rk == "inf" or not lv:
        return sign if negative else 0
    return rounded(lv/rv,dtype,negative_zero=negative)


def square_root(word,dtype):
    _,fraction,bias,inf,sign = fields(dtype)
    kind,value = decode(word,dtype)
    if kind == "nan" or int(word) & sign and (kind == "inf" or value != 0):
        return inf | 1 << (fraction-1)
    if kind == "inf" or not value:
        return int(word)
    exponent = max(log2_floor(value)//2,1-bias)
    scaled = value / Fraction(2)**(2*(exponent-fraction))
    root = isqrt(scaled.numerator//scaled.denominator)
    lhs,rhs = 4*scaled.numerator,scaled.denominator*(2*root+1)**2
    root += lhs > rhs or lhs == rhs and root % 2
    return pack(root,exponent,False,dtype)


def clamp(word,low,high,dtype,*,propagate):
    _,fraction,_,inf,sign = fields(dtype)
    kind,value = decode(word,dtype)
    if kind == "nan":
        return inf | 1 << (fraction-1) if propagate else int(low)
    def order(bits):
        # Monotonic total ordering of non-NaN binary float words, including
        # -0 < +0. Bounds in this reference must be defined and ordered.
        bits = int(bits)
        return ~bits if bits & sign else bits | sign
    assert decode(low,dtype)[0] != "nan" and decode(high,dtype)[0] != "nan"
    assert order(low) <= order(high) or decode(low,dtype)[1] == decode(high,dtype)[1] == 0
    return min(max(int(word),int(low),key=order),int(high),key=order)


def arrays(dtype,words):
    exponent,fraction = FLOAT_FORMATS[dtype]
    value = np.asarray(words,dtype=f"uint{1+exponent+fraction}")
    return value if dtype == "bf16" else value.view(DTYPES[dtype])


def reference(inputs,dtype):
    exponent,fraction = FLOAT_FORMATS[dtype]
    words = [value.view(f"uint{1+exponent+fraction}") for value in inputs]
    x,y,low,high = words
    output = [[],[],[],[]]
    for a,b,lo,hi in zip(x,y,low,high,strict=True):
        output[0].append(clamp(a,lo,hi,dtype,propagate=False))
        output[1].append(clamp(a,lo,hi,dtype,propagate=True))
        output[2].append(divide(a,b,dtype))
        output[3].append(square_root(a,dtype))
    return [arrays(dtype,value) for value in output]


def recurrence(inputs,dtype,steps,enabled):
    exponent,fraction = FLOAT_FORMATS[dtype]
    words,y = (value.view(f"uint{1+exponent+fraction}") for value in inputs[:2])
    current = [int(value) for value in words]
    for _ in range(steps):
        if enabled:
            current = [square_root(divide(a,b,dtype),dtype) for a,b in zip(current,y,strict=True)]
    return [arrays(dtype,current),*inputs[1:]]


def inputs(dtype,n=97):
    exponent,fraction,bias,inf,sign = fields(dtype)
    one = rounded(Fraction(1),dtype)
    special = [0,sign,1,sign|1,(1 << fraction)-1,1 << fraction,inf-1,sign|(inf-1),
        inf,sign|inf,inf|1,sign|inf|(1 << (fraction-1)),one,sign|one,one+1,one-1,
        rounded(Fraction(2),dtype),rounded(Fraction(3),dtype)]
    rng = np.random.default_rng(901)
    width = 1+exponent+fraction
    random = [int(value) & ((1 << width)-1) for value in rng.bit_generator.random_raw(79)]
    x = np.resize(np.asarray(special+random,dtype=f"uint{width}"),n)
    y = np.resize(np.asarray(list(reversed(special))+random[::-1],dtype=f"uint{width}"),n)
    low,high = np.full(n,sign|one,dtype=f"uint{width}"),np.full(n,one,dtype=f"uint{width}")
    if n > 0: low[0],high[0] = 0,0
    if n > 1: low[1],high[1] = sign,sign
    if n > 2: x[2],low[2],high[2] = sign,0,0
    if n > 3: x[3],high[3] = 0,sign
    return [arrays(dtype,value) for value in (x,y,low,high)]
