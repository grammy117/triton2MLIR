"""Pure SCF result facts require real initial/yield edges and checked headers."""

import json
import os

import numpy as np
import pytest
from xdsl.dialects.builtin import StringAttr, UnitAttr

from t2m.diagnostics import ConversionError
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, _functionalize, parse_shared


def projected_exit(width=32, *, initial="%lo", update="%next", slot=0, bounds=(1, 3), unsigned=False, double=False):
    second = f'''%end2 = scf.for %iv2 = %other to %hi step %step iter_args(%exit2 = %other) -> (i{width}) : i{width} {{
        %next2 = arith.addi %iv2, %step : i{width}
        scf.yield %next2 : i{width}
      }}
      %chosen = arith.addi %result#{slot}, %end2 : i{width}''' if double else ""
    chosen = "%chosen" if double else f"%result#{slot}"
    module = parse_shared(f'''builtin.module {{
      "tt.func"() <{{sym_name="kernel", function_type=(!tt.ptr<i64>, !tt.ptr<i64>, i{width}, i{width}, i{width}, i{width}) -> ()}}> ({{
      ^bb0(%x: !tt.ptr<i64>, %out: !tt.ptr<i64>, %lo: i{width}, %hi: i{width}, %step: i{width}, %other: i{width}):
        %result:2 = scf.for %iv = %lo to %hi step %step
            iter_args(%exit = {initial}, %decoy = %other) -> (i{width}, i{width}) : i{width} {{
          %next = arith.addi %iv, %step : i{width}
          %wrong = arith.addi %iv, %other : i{width}
          scf.yield {update}, %wrong : i{width}, i{width}
        }}
        {second}
        %offset = arith.index_cast {chosen} : i{width} to index
        %ptr = "tts.make_tptr"(%x, %offset) <{{operandSegmentSizes=array<i32: 1, 0, 1, 0>,
          sizes=array<i64: 1>, static_offsets=array<i64: -9223372036854775808>, static_strides=array<i64: 1>,
          static_shape=array<i64: 0>, order=array<i32>}}> : (!tt.ptr<i64>, index) -> tensor<1x!tt.ptr<i64>>
        %value = "tts.load"(%ptr) <{{operandSegmentSizes=array<i32: 1, 0, 0>, static_mask_dims=array<i64>}}> :
          (tensor<1x!tt.ptr<i64>>) -> tensor<1xi64>
        %dest = "tts.make_tptr"(%out) <{{operandSegmentSizes=array<i32: 1, 0, 0, 0>, sizes=array<i64: 1>,
          static_offsets=array<i64: 0>, static_strides=array<i64: 1>, static_shape=array<i64: 0>, order=array<i32>}}> :
          (!tt.ptr<i64>) -> tensor<1x!tt.ptr<i64>>
        "tts.store"(%dest, %value) <{{static_mask_dims=array<i64>}}> : (tensor<1x!tt.ptr<i64>>, tensor<1xi64>) -> ()
        "tt.return"() : () -> ()
      }}) : () -> ()
    }}''')
    loop = next(op for op in module.walk() if op.name == "scf.for")
    loop.attributes["prov.exit_induction"] = StringAttr("claimed")
    if unsigned:
        loop.attributes["unsignedCmp"] = UnitAttr()
    abi = TensorABI("kernel", (TensorBuffer("x", (32,), "read", "i64"), TensorBuffer("out", (1,), "write", "i64")),
        runtime_scalars=tuple(RuntimeScalar(name, f"i{width}", index, bounds=interval)
            for name, index, interval in (("lo", 2, (0, 3)), ("hi", 3, (0, 8)), ("step", 4, bounds), ("other", 5, (1, 3)))))
    return _functionalize(module, abi)


@pytest.mark.parametrize("width", [32, 64])
def test_pure_loop_keeps_actual_headers_complete_tuple_and_only_the_proved_result_fact(width):
    result = projected_exit(width)
    loop = next(op for op in result.walk() if op.name == "scf.for")
    function = next(iter(result.ops))
    entry = function.body.block
    assert tuple(loop.operands[:3]) == tuple(entry.args[1:4])
    assert len(loop.results) == 2
    assert json.loads(function.attributes["prov.source_loops"].data)[0]["actual_exit_induction_results"] == [0]
    result.verify()


@pytest.mark.parametrize("changes", [{"initial": "%other"}, {"update": "%wrong"}, {"slot": 1},
    {"bounds": None}, {"bounds": (0, 3)}, {"unsigned": True}])
def test_annotations_equal_intervals_and_unproved_source_edges_do_not_authorize_exit_addresses(changes):
    with pytest.raises(ConversionError):
        projected_exit(**changes)


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native environment required")
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("double", [False, True])
def test_native_final_reads_use_actual_exit_and_keep_nonzero_zero_trip_initial_values(tmp_path, width, double):
    from t2m.ir import text
    from t2m.native import execute_native

    result = projected_exit(width, double=double)
    x = np.arange(32, dtype=np.int64) * 37 + (1 << 55)
    integer = np.int32 if width == 32 else np.int64
    for index, (lo, hi, stride) in enumerate(((1, 8, 3), (3, 1, 2), (0, 0, 1))):
        end = lo + len(range(lo, hi, stride)) * stride
        if double:
            end += 2 + len(range(2, hi, stride)) * stride
        arrays = [x, *[np.asarray(value, integer) for value in (lo, hi, stride, 2)]]
        (actual,), _ = execute_native(text(result), arrays, tmp_path / f"native-{index}", python=os.environ["T2M_NATIVE_PYTHON"])
        assert (actual.shape, actual.dtype, actual.tobytes()) == ((1,), x.dtype, x[end:end+1].tobytes())
