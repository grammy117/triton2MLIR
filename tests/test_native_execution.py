"""Actual LLVM JIT checks, separate from compiler transport and IR verification."""

import os

import numpy as np
import pytest

from t2m.native import execute_native

NATIVE = os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not NATIVE, reason="existing native environment not configured")

SLICE = """module {
  func.func @slice(%x: tensor<4x6xf32>) -> tensor<2x3xf32> {
    %0 = tensor.extract_slice %x[1, 2] [2, 3] [1, 1] : tensor<4x6xf32> to tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}"""


def test_native_strided_copy_executes_serialized_ir(tmp_path):
    x = np.arange(24, dtype=np.float32).reshape(4, 6)
    before = x.copy()
    (actual,), report = execute_native(SLICE, [x], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual, x[1:3, 2:5])
    np.testing.assert_array_equal(x, before)
    assert report["execution"] == "native-host-LLVM-JIT"
    assert not report["source_triton_executed"]
    assert report["consumer_execution"] == "not-performed"


def test_native_multiple_results_keep_order(tmp_path):
    source = """module {
      func.func @two(%x: tensor<4x6xf32>, %y: tensor<4x6xf32>) -> (tensor<4x6xf32>, tensor<4x6xf32>) {
        return %y, %x : tensor<4x6xf32>, tensor<4x6xf32>
      }
    }"""
    x, y = np.zeros((4, 6), np.float32), np.ones((4, 6), np.float32)
    actual, _ = execute_native(source, [x, y], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual[0], y)
    np.testing.assert_array_equal(actual[1], x)


def test_native_rejects_wrong_input_type_and_unknown_ops(tmp_path):
    with pytest.raises(ValueError, match="shape/dtype"):
        execute_native(SLICE, [np.zeros((4, 6), np.float64)], tmp_path / "wrong", python=NATIVE)
    with pytest.raises(ValueError, match="native host execution failed"):
        execute_native('"unqualified.magic"() : () -> ()', [], tmp_path / "unknown", python=NATIVE)


def test_native_alias_precondition_is_checked_before_launch(tmp_path):
    x = np.zeros((4, 6), np.float32)
    with pytest.raises(ValueError, match="independently allocated"):
        execute_native(SLICE, [x, x[:, :]], tmp_path / "alias", python=NATIVE)
    assert not (tmp_path / "alias").exists()


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_standard_reciprocal_roots_use_executable_expansion_without_nonstandard_libm(tmp_path, dtype):
    from t2m.examples.scalar_math_reference import divide, square_root
    from t2m.tensor_storage import assert_ieee_words

    dt, integer = (np.float32, np.uint32) if dtype == "f32" else (np.float64, np.uint64)
    values = np.asarray([0., -0., np.nextafter(dt(0), dt(1)), 1., 2., np.inf, -np.inf, -1., np.nan], dt)
    source = f'''module {{func.func @roots(%x:tensor<9x{dtype}>)->tensor<9x{dtype}> {{
      %e=tensor.empty():tensor<9x{dtype}>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%x:tensor<9x{dtype}>) outs(%e:tensor<9x{dtype}>) {{
        ^bb0(%a:{dtype},%unused:{dtype}):
          %v=math.rsqrt %a:{dtype}
          linalg.yield %v:{dtype}
        }}->tensor<9x{dtype}>
      return %r:tensor<9x{dtype}>
    }}}}'''
    (actual,), report = execute_native(source,[values],tmp_path / "native",python=NATIVE)
    one = int(np.asarray(1.,dt).view(integer))
    expected = np.asarray([divide(one,square_root(int(word),dtype),dtype) for word in values.view(integer)],integer).view(dt)
    assert_ieee_words(actual,expected,dtype)
    assert any("math-expand-ops{ops=rsqrt},convert-math-to-libm" in pipeline for pipeline in report["lowering_pipelines"])
