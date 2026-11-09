import os

import numpy as np
import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr, i32

from t2m.frontend import capture_ttir
from t2m.diagnostics import ConversionError
from t2m.native import execute_native
from t2m.shared_tensor import TensorABI, TensorBuffer, _functionalize, convert_shared, parse_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared optimizer and native environment required")


@pytest.mark.parametrize("store", [False, True])
def test_transposed_structured_accesses_preserve_tensor_values(tmp_path, store):
    from t2m.examples.tensor_kernels import transposed_loop_matmul, transpose_store
    if store:
        kernel, constants, grid = transpose_store, {"M": 4, "N": 8}, (1,)
        buffers = (TensorBuffer("x", (4, 8)), TensorBuffer("out", (8, 4), "write"))
    else:
        kernel, grid = transposed_loop_matmul, (2, 3)
        constants = {"N": 48, "K": 32, "BM": 16, "BN": 16, "BK": 16}
        buffers = (TensorBuffer("x", (32, 32)), TensorBuffer("weight", (48, 32)), TensorBuffer("out", (32, 48), "write"))
    source = tmp_path / "kernel.ttir"
    source.write_text(capture_ttir(kernel, {**{b.name: "*fp32" for b in buffers},
                                          **{n: "constexpr" for n in constants}}, constants))
    result = convert_shared(source, TensorABI(kernel.__name__, buffers, grid=grid), tmp_path / "translation", optimizer=OPT)
    assert any(op.name == "linalg.transpose" for op in result.module.walk())
    rng = np.random.default_rng(25)
    arrays = [rng.normal(size=b.shape).astype("float32") for b in result.spec.inputs]
    (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE)
    expected = arrays[0].T if store else arrays[0] @ arrays[1].T
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=5e-6)


@pytest.mark.parametrize("shape", [(1, 8), (4, 1), (1, 1)])
def test_zero_structured_strides_are_valid_only_on_unit_source_axes(tmp_path, shape):
    from t2m.examples.tensor_kernels import matrix_add
    constants = {"M": shape[0], "N": shape[1]}
    buffers = (TensorBuffer("x", shape), TensorBuffer("y", shape), TensorBuffer("out", shape, "write"))
    source = tmp_path / "kernel.ttir"
    source.write_text(capture_ttir(matrix_add, {**{b.name: "*fp32" for b in buffers},
                                               **{n: "constexpr" for n in constants}}, constants))
    result = convert_shared(source, TensorABI("matrix_add", buffers), tmp_path / "translation", optimizer=OPT)
    x = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    y = np.full(shape, 3, dtype="float32")
    (actual,), _ = execute_native(result.mlir_text, [x, y], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual, x + y)
    if shape == (1, 1):
        invalid = parse_shared((tmp_path / "translation/tensor-ir/lowered.mlir").read_text())
        index = next(op for op in invalid.walk() if isinstance(op, arith.ConstantOp)
                     and isinstance(op.value, IntegerAttr) and op.value.type == i32)
        index.properties["value"] = IntegerAttr(1, i32)
        with pytest.raises(ConversionError, match="unit gather/scatter index"):
            _functionalize(invalid, result.spec)
