"""Logical tensor types and their explicit NumPy archive/storage carriers.

BF16 tensors retain bf16 in MLIR. NumPy carries their raw two-byte words as
uint16, matching Merlin's host descriptor storage. Decode only for numerical
comparison; never reinterpret those words as integer source arithmetic.
FP8 tensors similarly retain their format and use raw uint8 archive words.
"""

import numpy as np

DTYPES = {"f16": np.float16, "bf16": np.uint16, "f32": np.float32, "f64": np.float64,
          "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64, "i1": np.bool_}
FP8_NUMPY_NAMES = {"f8E4M3FN":"float8_e4m3fn", "f8E4M3FNUZ":"float8_e4m3fnuz",
                   "f8E5M2":"float8_e5m2", "f8E5M2FNUZ":"float8_e5m2fnuz"}
DTYPES.update({name: np.uint8 for name in FP8_NUMPY_NAMES})

# Standard binary interchange formats, independent of any execution target.
FLOAT_FORMATS = {"f16":(5,10),"bf16":(8,7),"f32":(8,23),"f64":(11,52)}


def comparison_values(array, dtype):
    array = np.asarray(array)
    if dtype in FP8_NUMPY_NAMES:
        if array.dtype != np.uint8:
            raise ValueError("FP8 archive storage requires raw uint8 words")
        import ml_dtypes
        return array.view(getattr(ml_dtypes, FP8_NUMPY_NAMES[dtype])).astype(np.float32)
    if dtype != "bf16":
        return array
    if array.dtype != np.uint16:
        raise ValueError("BF16 archive storage requires raw uint16 words")
    return (array.astype(np.uint32) << 16).view(np.float32)


def assert_tensor_close(actual, expected, dtype, *, rtol=0, atol=0):
    """Compare exact integer values without NumPy's lossy floating promotion."""
    left, right = comparison_values(actual, dtype), comparison_values(expected, dtype)
    if dtype.startswith("i") and rtol == 0 and atol == 0:
        np.testing.assert_array_equal(left, right)
    else:
        np.testing.assert_allclose(left, right, rtol=rtol, atol=atol, equal_nan=True)


def assert_ieee_words(actual,expected,dtype):
    """Exact finite/infinite/zero words; NaN class without payload selection."""
    if dtype not in FLOAT_FORMATS:
        raise ValueError("IEEE result comparison requires a declared floating tensor type")
    exponent,fraction = FLOAT_FORMATS[dtype]
    width = 1+exponent+fraction
    left,right = (np.asarray(value).view(f"uint{width}") for value in (actual,expected))
    fraction_mask,exponent_mask = (1 << fraction)-1,((1 << exponent)-1) << fraction
    def isnan(words):
        return ((words & exponent_mask) == exponent_mask) & ((words & fraction_mask) != 0)
    left_nan,right_nan = isnan(left),isnan(right)
    np.testing.assert_array_equal(left_nan,right_nan)
    np.testing.assert_array_equal(left[~left_nan],right[~right_nan])
