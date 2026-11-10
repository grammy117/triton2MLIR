"""Logical tensor types and their explicit NumPy archive/storage carriers.

BF16 tensors retain bf16 in MLIR. NumPy carries their raw two-byte words as
uint16, matching Merlin's host descriptor storage. Decode only for numerical
comparison; never reinterpret those words as integer source arithmetic.
"""

import numpy as np

DTYPES = {"f16": np.float16, "bf16": np.uint16, "f32": np.float32, "f64": np.float64,
          "i8": np.int8, "i16": np.int16, "i32": np.int32, "i64": np.int64, "i1": np.bool_}


def comparison_values(array, dtype):
    array = np.asarray(array)
    if dtype != "bf16":
        return array
    if array.dtype != np.uint16:
        raise ValueError("BF16 archive storage requires raw uint16 words")
    return (array.astype(np.uint32) << 16).view(np.float32)
