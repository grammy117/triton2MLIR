"""Exact retained unit axes and omitted-axis bounds in readonly tile slices."""

import os

import numpy as np
import pytest
from xdsl.dialects.builtin import DYNAMIC_INDEX

from t2m.diagnostics import ConversionError
from t2m.native import execute_native
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_indexing import Linear, logical_coordinates
from t2m.shared_tensor import TensorABI, TensorBuffer, _functionalize, parse_shared
from t2m.views import axis_permutation


def projected(bounds=(0, 16), *, effect="read", masked=False):
    mask = "0" if masked else ""
    module = parse_shared('''"builtin.module"() ({
      "tt.func"() <{sym_name = "kernel", function_type = (!tt.ptr<i64>, !tt.ptr<i64>, i32) -> ()}> ({
      ^bb0(%x: !tt.ptr<i64>, %out: !tt.ptr<i64>, %row: i32):
        %i = "arith.index_cast"(%row) : (i32) -> index
        %eight = "arith.constant"() <{value = 8 : index}> : () -> index
        %offset = "arith.muli"(%i, %eight) : (index, index) -> index
        %ptr = "tts.make_tptr"(%x, %offset) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>,
          sizes = array<i64: 1>, static_offsets = array<i64: -9223372036854775808>,
          static_strides = array<i64: 8>, static_shape = array<i64: 0>, order = array<i32>}> :
          (!tt.ptr<i64>, index) -> tensor<1x!tt.ptr<i64>>
        %value = "tts.load"(%ptr) <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64MASK>}> : (tensor<1x!tt.ptr<i64>>) -> tensor<1xi64>
        %dest = "tts.make_tptr"(%out) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, sizes = array<i64: 1>,
          static_offsets = array<i64: 0>, static_strides = array<i64: 1>, static_shape = array<i64: 0>,
          order = array<i32>}> : (!tt.ptr<i64>) -> tensor<1x!tt.ptr<i64>>
        "tts.store"(%dest, %value) <{static_mask_dims = array<i64>}> :
          (tensor<1x!tt.ptr<i64>>, tensor<1xi64>) -> ()
        "tt.return"() : () -> ()
      }) : () -> ()
    }) : () -> ()'''.replace("i64MASK", "i64: " + mask if mask else "i64"))
    abi = TensorABI("kernel", (TensorBuffer("x", (17, 8), effect, "i64"), TensorBuffer("out", (1,), "write", "i64")),
        runtime_scalars=(RuntimeScalar("row", "i32", 2, bounds=bounds),))
    return _functionalize(module, abi), abi


def test_retained_unit_axis_uses_an_actual_one_size_operand():
    module, _abi = projected()
    operation = next(op for op in module.walk() if op.name == "tensor.extract_slice"
                     and len(op.operands[0].type.get_shape()) == 2)
    assert tuple(operation.static_sizes.get_values()) == (DYNAMIC_INDEX, 1)
    assert operation.results[0].type.get_shape() == (DYNAMIC_INDEX,)
    assert len(operation.sizes) == 1 and operation.sizes[0].owner.value.value.data == 1


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native environment required")
def test_native_unit_projection_preserves_the_selected_root_coordinate(tmp_path):
    from t2m.ir import text

    module, abi = projected()
    x = np.arange(17 * 8, dtype=np.int64).reshape(17, 8) + (1 << 55)
    for row in (0, 7, 16):
        actual, _ = execute_native(text(module), [x, np.asarray(row, np.int32)], tmp_path / f"native-{row}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=abi.runtime_shapes())
        assert actual[0].tobytes() == x[row, :1].tobytes()


@pytest.mark.parametrize("options", [{"bounds": (0, 17)}, {"effect": "readwrite"}, {"masked": True}])
def test_projection_does_not_grant_outside_axis_writable_or_undefined_fill_permission(options):
    with pytest.raises(ConversionError):
        projected(**options)


def test_projected_axis_selection_is_injective_and_explicit():
    buffer = TensorBuffer("x", (3, 5, 7))
    offsets = (Linear(1), Linear(2), Linear(0))
    assert axis_permutation(buffer, (1, 35), offsets, (7, 1), (1,), physical_offsets=True,
        allow_rank_reduction=True) == (2, 0)
    with pytest.raises(ConversionError):
        axis_permutation(buffer, (1, 1), offsets, (7, 7), (1,), physical_offsets=True, allow_rank_reduction=True)
    with pytest.raises(ConversionError):
        axis_permutation(buffer, (1,), offsets, (7,), (1,), physical_offsets=True)


def test_logical_coordinates_preserve_scalar_parameter_terms_and_loop_terms():
    total = Linear(1, (7, 0, 0), ((0, 0, 2, 35),), (("row", 0, 1, 70),))
    coordinates = logical_coordinates(total, (35, 7, 1), (3,))
    assert coordinates == (Linear(loops=((0, 0, 2, 1),), parameters=(("row", 0, 1, 2),)),
                           Linear(programs=(1, 0, 0)), Linear(1))
    restored = Linear()
    for coordinate, stride in zip(coordinates, (35, 7, 1), strict=True):
        restored += coordinate.scale(stride)
    assert restored == total


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native environment required")
def test_projected_multiaxis_tile_retains_its_original_transposed_order(tmp_path):
    from t2m.ir import text

    source = parse_shared('''"builtin.module"() ({
      "tt.func"() <{sym_name = "kernel", function_type = (!tt.ptr<i64>, !tt.ptr<i64>, i32) -> ()}> ({
      ^bb0(%x: !tt.ptr<i64>, %out: !tt.ptr<i64>, %row: i32):
        %i = "arith.index_cast"(%row) : (i32) -> index
        %seven = "arith.constant"() <{value = 7 : index}> : () -> index
        %offset = "arith.muli"(%i, %seven) : (index, index) -> index
        %ptr = "tts.make_tptr"(%x, %offset) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>,
          sizes = array<i64: 7, 3>, static_offsets = array<i64: -9223372036854775808, 0>,
          static_strides = array<i64: 1, 35>, static_shape = array<i64: 0, 0>, order = array<i32>}> :
          (!tt.ptr<i64>, index) -> tensor<7x3x!tt.ptr<i64>>
        %value = "tts.load"(%ptr) <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}> : (tensor<7x3x!tt.ptr<i64>>) -> tensor<7x3xi64>
        %dest = "tts.make_tptr"(%out) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, sizes = array<i64: 7, 3>,
          static_offsets = array<i64: 0, 0>, static_strides = array<i64: 3, 1>, static_shape = array<i64: 0, 0>,
          order = array<i32>}> : (!tt.ptr<i64>) -> tensor<7x3x!tt.ptr<i64>>
        "tts.store"(%dest, %value) <{static_mask_dims = array<i64>}> :
          (tensor<7x3x!tt.ptr<i64>>, tensor<7x3xi64>) -> ()
        "tt.return"() : () -> ()
      }) : () -> ()
    }) : () -> ()''')
    abi = TensorABI("kernel", (TensorBuffer("x", (3, 5, 7), "read", "i64"), TensorBuffer("out", (7, 3), "write", "i64")),
        runtime_scalars=(RuntimeScalar("row", "i32", 2, bounds=(0, 4)),))
    module = _functionalize(source, abi)
    assert any(op.name == "linalg.transpose" for op in module.walk())
    x = np.arange(3 * 5 * 7, dtype=np.int64).reshape(3, 5, 7) + (1 << 55)
    for row in (0, 2, 4):
        actual, _ = execute_native(text(module), [x, np.asarray(row, np.int32)], tmp_path / f"native-{row}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=abi.runtime_shapes())
        expected = x[:, row, :].T.copy()
        assert (actual[0].shape, actual[0].dtype, actual[0].tobytes()) == (expected.shape, expected.dtype, expected.tobytes())
