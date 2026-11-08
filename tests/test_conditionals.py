import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.native import execute_native
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared optimizer and native environment required")


def convert(kernel, root, effect="write", size=16):
    source = root / "kernel.ttir"
    source.write_text(capture_ttir(kernel, {"x": "*fp32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", (size,)), TensorBuffer("out", (size,), effect)), grid=(size // 8,))
    return convert_shared(source, abi, root / "translation", optimizer=OPT)


def test_data_dependent_branches_merge_fresh_tensor_results(tmp_path):
    from t2m.examples.tensor_kernels import conditional_update
    result = convert(conditional_update, tmp_path)
    assert any(op.name == "scf.if" for op in result.module.walk())
    x = np.concatenate([np.ones(8, np.float32), -np.ones(8, np.float32)])
    (actual,), _ = execute_native(result.mlir_text, [x], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual, np.concatenate([x[:8] + 3, x[8:] * 2]))


def test_missing_else_preserves_incoming_values_and_cannot_initialize_fresh_output(tmp_path):
    from t2m.examples.tensor_kernels import conditional_partial
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with pytest.raises(ConversionError, match="complete overwrite"):
        convert(conditional_partial, fresh)
    updated = tmp_path / "updated"
    updated.mkdir()
    result = convert(conditional_partial, updated, effect="readwrite")
    x = np.concatenate([np.ones(8, np.float32), -np.ones(8, np.float32)])
    initial = np.arange(16, dtype=np.float32)
    (actual,), _ = execute_native(result.mlir_text, [x, initial], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual, np.concatenate([x[:8] + 3, initial[8:]]))


def test_store_in_one_branch_does_not_initialize_the_other_branch(tmp_path):
    from t2m.examples.tensor_kernels import conditional_uninitialized
    with pytest.raises(ConversionError, match="read before.*initialized"):
        convert(conditional_uninitialized, tmp_path, size=8)


def test_branch_values_and_memory_versions_survive_nested_loop(tmp_path):
    from t2m.examples.tensor_kernels import loop_conditional_update
    result = convert(loop_conditional_update, tmp_path)
    x = np.concatenate([np.ones(8, np.float32), -np.ones(8, np.float32)])
    expected = x.copy()
    for _ in range(3):
        expected[:8] = expected[:8] + 2
        expected[8:] = expected[8:] * 2 + 1
    (actual,), _ = execute_native(result.mlir_text, [x], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual, expected)
    assert result.trace["memory"]["source_conditionals"][0]["initialization"] == "intersection-of-both-branch-facts"
