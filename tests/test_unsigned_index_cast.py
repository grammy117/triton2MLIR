"""Standard unsigned index casts retain bits and semantic flags."""

import os

import numpy as np
import pytest
from xdsl.utils.exceptions import VerifyException

from t2m.ir import text
from t2m.shared_tensor import parse_shared
from t2m.standard_arith import IndexCastUIOp


@pytest.mark.parametrize("flag", ["", "nneg"])
def test_unsigned_index_cast_roundtrip_keeps_source_promise_and_annotations(flag):
    source = f'''module {{func.func @forward(%a:i32)->i64 {{
      %i=arith.index_castui %a {flag} {{prov.op="shape_guard"}} : i32 to index
      %v=arith.index_castui %i : index to i64
      return %v:i64
    }}}}'''
    module = parse_shared(source)
    module.verify()
    operations = [op for op in module.walk() if isinstance(op, IndexCastUIOp)]
    assert len(operations) == 2 and (operations[0].nonNeg is not None) == bool(flag)
    again = parse_shared(text(module))
    again.verify()
    assert text(again) == text(module)


@pytest.mark.parametrize("source,target", [("index", "index"), ("i32", "i64"), ("f32", "index"), ("i0", "index")])
def test_invalid_unsigned_index_cast_does_not_get_generic_unknown_admission(source, target):
    module = parse_shared(f'''module {{func.func @forward(%a:{source})->{target} {{
      %v=arith.index_castui %a : {source} to {target}
      return %v:{target}
    }}}}''')
    with pytest.raises(VerifyException):
        module.verify()


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native toolchain required")
def test_native_unsigned_index_widening_retains_high_source_bits(tmp_path):
    from t2m.native import execute_native

    source = '''module {func.func @forward(%x:tensor<4xi32>)->tensor<4xi64> {
      %empty=tensor.empty():tensor<4xi64>
      %out=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
        iterator_types=["parallel"]} ins(%x:tensor<4xi32>) outs(%empty:tensor<4xi64>) {
      ^bb0(%word:i32,%old:i64):
        %index=arith.index_castui %word : i32 to index
        %wide=arith.index_castui %index : index to i64
        linalg.yield %wide:i64
      }->tensor<4xi64>
      return %out:tensor<4xi64>
    }}'''
    parse_shared(source).verify()
    words = np.asarray([0, -1, -(1 << 31), 17], dtype=np.int32)
    (actual,), _ = execute_native(source, [words], tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    np.testing.assert_array_equal(actual, np.asarray([0, (1 << 32) - 1, 1 << 31, 17], dtype=np.int64))
