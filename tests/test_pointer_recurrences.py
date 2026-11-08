import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.native import execute_native
from t2m.shared_tensor import TensorABI, TensorBuffer, _functionalize, convert_shared, parse_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared optimizer and native environment required")


def test_pointer_increment_gemm_uses_upstream_index_recurrences(tmp_path):
    from t2m.examples.tensor_kernels import pointer_loop_matmul
    constants = {"N": 32, "K": 48, "BM": 16, "BN": 16, "BK": 16}
    buffers = (TensorBuffer("x", (32, 48)), TensorBuffer("y", (48, 32)), TensorBuffer("out", (32, 32), "write"))
    source = tmp_path / "kernel.ttir"
    source.write_text(capture_ttir(pointer_loop_matmul, {**{b.name: "*fp32" for b in buffers},
                                                      **{n: "constexpr" for n in constants}}, constants))
    result = convert_shared(source, TensorABI("pointer_loop_matmul", buffers, grid=(2, 2)), tmp_path / "translation", optimizer=OPT)
    recurrences = result.trace["memory"]["source_loops"][0]["index_recurrences"]
    assert sorted(r["increment"] for r in recurrences) == [16, 512]
    rng = np.random.default_rng(26)
    x, y = [rng.normal(size=b.shape).astype("float32") for b in buffers[:2]]
    (actual,), _ = execute_native(result.mlir_text, [x, y], tmp_path / "native", python=NATIVE)
    np.testing.assert_allclose(actual, x @ y, rtol=1e-5, atol=5e-6)


def test_nonunit_loop_header_and_final_pointer_value_are_preserved(tmp_path):
    from t2m.examples.tensor_kernels import pointer_loop_after
    source = tmp_path / "kernel.ttir"
    source.write_text(capture_ttir(pointer_loop_after, {"x": "*fp32", "out": "*fp32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("pointer_loop_after", (TensorBuffer("x", (32,)), TensorBuffer("out", (8,), "write")))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT)
    x = np.arange(32, dtype=np.float32)
    (actual,), _ = execute_native(result.mlir_text, [x], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual, x.reshape(4, 8).sum(axis=0))
    assert result.trace["memory"]["source_loops"][0]["iterations"] == 3
    small = TensorABI(abi.entry, (TensorBuffer("x", (24,)), abi.buffers[1]))
    with pytest.raises(ConversionError, match="bounds"):
        convert_shared(source, small, tmp_path / "out-of-bounds", optimizer=OPT)


def test_observable_poison_pointer_is_never_replaced_with_an_initializer():
    module = parse_shared('''"builtin.module"() ({
      "tt.func"() <{sym_name = "kernel", function_type = (!tt.ptr<f32>) -> ()}> ({
      ^bb0(%out: !tt.ptr<f32>):
        %pointer = "ub.poison"() <{value = #ub.poison}> : () -> tensor<8x!tt.ptr<f32>>
        %value = "tt.load"(%pointer) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<8x!tt.ptr<f32>>) -> tensor<8xf32>
        "tt.return"() : () -> ()
      }) : () -> ()
    }) : () -> ()''')
    with pytest.raises(ConversionError, match="poison is observable"):
        _functionalize(module, TensorABI("kernel", (TensorBuffer("out", (8,), "readwrite"),)))
