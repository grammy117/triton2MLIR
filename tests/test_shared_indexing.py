import pytest

from t2m.diagnostics import ConversionError
from t2m.shared_indexing import check_source_addresses
from t2m.shared_tensor import TensorABI, TensorBuffer, parse_shared


def test_source_integer_overflow_is_checked_before_pointer_reraising():
    source = '''"builtin.module"() ({
      "tt.func"() <{sym_name = "kernel", function_type = (!tt.ptr<f32>, i32) -> ()}> ({
      ^bb0(%p: !tt.ptr<f32>, %stride: i32):
        %i = "tt.make_range"() <{start = 0 : i32, end = 4 : i32}> : () -> tensor<4xi32>
        %s = "tt.splat"(%stride) : (i32) -> tensor<4xi32>
        %offset = "arith.muli"(%i, %s) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
        %base = "tt.splat"(%p) : (!tt.ptr<f32>) -> tensor<4x!tt.ptr<f32>>
        %ptr = "tt.addptr"(%base, %offset) : (tensor<4x!tt.ptr<f32>>, tensor<4xi32>) -> tensor<4x!tt.ptr<f32>>
        "tt.return"() : () -> ()
      }) : () -> ()
    }) : () -> ()'''
    module = parse_shared(source)
    buffer = TensorBuffer("out", (4,), "readwrite")
    check_source_addresses(module, TensorABI("kernel", (buffer,), scalar_constants=(2,)))
    with pytest.raises(ConversionError, match="overflow before upstream index promotion"):
        check_source_addresses(module, TensorABI("kernel", (buffer,), scalar_constants=(2**30,)))
