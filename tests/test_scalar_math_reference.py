"""Known IEEE vectors and exact midpoint boundaries test the integer oracle."""

from fractions import Fraction

import pytest

from t2m.examples.scalar_math_reference import divide,square_root,rounded,fields


@pytest.mark.parametrize("dtype,one,three,third,sqrt_two",[
    ("f16",0x3c00,0x4200,0x3555,0x3da8),
    ("bf16",0x3f80,0x4040,0x3eab,0x3fb5),
    ("f32",0x3f800000,0x40400000,0x3eaaaaab,0x3fb504f3),
    ("f64",0x3ff0000000000000,0x4008000000000000,0x3fd5555555555555,0x3ff6a09e667f3bcd),
])
def test_exact_known_vectors_and_midpoints(dtype,one,three,third,sqrt_two):
    assert divide(one,three,dtype) == third
    assert square_root(rounded(Fraction(2),dtype),dtype) == sqrt_two
    assert square_root(rounded(Fraction(4),dtype),dtype) == rounded(Fraction(2),dtype)
    _,fraction,bias,inf,sign = fields(dtype)
    assert rounded(Fraction(1)+Fraction(1,2**(fraction+1)),dtype) == one
    assert rounded(Fraction(1)+Fraction(3,2**(fraction+1)),dtype) == one+2
    minimum = Fraction(2)**(1-bias-fraction)
    assert rounded(minimum/2,dtype) == 0
    assert rounded(minimum*Fraction(3,2),dtype) == 2
    assert rounded(-minimum/2,dtype) == sign
    overflow = (Fraction(2)-Fraction(1,2**(fraction+1)))*Fraction(2)**bias
    assert rounded(overflow,dtype) == inf
    assert divide(one,0,dtype) == inf
    assert divide(sign,one,dtype) == sign
    assert square_root(sign,dtype) == sign
    assert decode_nan(divide(0,0,dtype),dtype)
    assert decode_nan(square_root(sign|one,dtype),dtype)


def decode_nan(word,dtype):
    from t2m.examples.scalar_math_reference import decode
    return decode(word,dtype)[0] == "nan"
