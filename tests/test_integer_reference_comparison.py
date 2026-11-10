"""Default exact integer validation must see low bits beyond binary64 precision."""

import numpy as np
import pytest

from t2m.tensor_storage import assert_tensor_close


@pytest.mark.parametrize("expected,actual", [(2**60, 2**60 + 1), (-(2**60), -(2**60) - 1),
    (np.iinfo(np.int64).max, np.iinfo(np.int64).max - 1), (np.iinfo(np.int64).min, np.iinfo(np.int64).min + 1)])
def test_zero_tolerance_i64_comparison_detects_individual_source_bits(expected, actual):
    arrays = [np.array([number], np.int64) for number in (actual, expected)]
    with pytest.raises(AssertionError):
        assert_tensor_close(*arrays, "i64")
    assert_tensor_close(arrays[1], arrays[1].copy(), "i64")


def test_bfloat_and_floating_comparisons_keep_their_declared_numeric_policy():
    assert_tensor_close(np.array([0x8000, 0x7f81], np.uint16), np.array([0, 0x7fc2], np.uint16), "bf16")
    assert_tensor_close(np.array([1.001], np.float32), np.array([1.], np.float32), "f32", rtol=.01)
