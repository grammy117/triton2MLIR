"""Checked Boolean byte observations retain logical dataflow and source IDs."""

import pytest
import os
import numpy as np

from t2m.boolean_memory import normalize_boolean_memory
from t2m.diagnostics import ConversionError
from t2m.shared_tensor import _source_ids, opcode, parse_shared


def fixture(*, fill=0, observation="nonzero", store="extended"):
    comparison = {'nonzero': '"arith.cmpi"(%v, %zero) <{predicate=1:i64}> : (i8,i8)->i1',
        'equal': '"arith.cmpi"(%v, %zero) <{predicate=0:i64}> : (i8,i8)->i1',
        'raw': '"arith.trunci"(%v) : (i8)->i1'}[observation]
    value = "%wide" if store == "extended" else "%raw"
    return parse_shared('''builtin.module { "tt.func"() <{sym_name="boolean", function_type=(!tt.ptr<i1>, i1, i8)->()}> ({
      ^bb0(%p:!tt.ptr<i1>, %mask:i1, %raw:i8):
        %byte = "tt.bitcast"(%p) : (!tt.ptr<i1>)->!tt.ptr<i8> loc("ttir.op.cast")
        %zero = arith.constant 0:i8
        %fill = arith.constant FILL:i8
        %v = "tt.load"(%byte, %mask, %fill) <{operandSegmentSizes=array<i32:1,1,1>}> : (!tt.ptr<i8>,i1,i8)->i8 loc("ttir.op.load")
        %bool = COMPARISON loc("ttir.op.compare")
        %wide = arith.extui %bool:i1 to i8 loc("ttir.op.extend")
        "tt.store"(%byte, VALUE, %mask) : (!tt.ptr<i8>,i8,i1)->() loc("ttir.op.store")
        "tt.return"() : ()->()
    }) : ()->() }'''.replace("FILL", str(fill)).replace("COMPARISON", comparison).replace("VALUE", value))


@pytest.mark.parametrize("fill", [0, 1, 2, -128])
def test_masks_nonzero_fills_and_original_source_correspondence(fill):
    module = fixture(fill=fill)
    reports = normalize_boolean_memory(module, _source_ids)
    module.verify()
    loads = [op for op in module.walk() if opcode(op) == "tt.load"]
    stores = [op for op in module.walk() if opcode(op) == "tt.store"]
    assert len(loads) == len(stores) == 1
    load, store = loads[0], stores[0]
    assert str(load.results[0].type) == str(store.operands[1].type) == "i1"
    assert load.operands[1] == store.operands[2]
    assert bool(load.operands[2].owner.value.value.data) == (fill != 0)
    assert any({"ttir.op.cast", "ttir.op.load", "ttir.op.compare"} <= set(r["source_ids"]) for r in reports)
    assert any({"ttir.op.cast", "ttir.op.store", "ttir.op.extend"} <= set(r["source_ids"]) for r in reports)
    assert not any(opcode(op) in ("tt.bitcast", "arith.cmpi") for op in module.walk())


@pytest.mark.parametrize("kwargs", [{"observation":"equal"}, {"observation":"raw"}, {"store":"raw"}])
def test_byte_observations_require_their_own_storage_abi_before_any_edits(kwargs):
    module = fixture(**kwargs)
    original = str(module)
    with pytest.raises(ConversionError) as error:
        normalize_boolean_memory(module, _source_ids)
    assert error.value.source_id and error.value.opcode
    assert str(module) == original


@pytest.mark.skipif(not (os.environ.get("TRITON_SHARED_OPT") and os.environ.get("T2M_NATIVE_PYTHON")), reason="existing native toolchain required")
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
@pytest.mark.parametrize("fill", [False, True])
def test_boolean_runtime_tails_masks_and_nonzero_fills(tmp_path, profile, fill):
    from t2m.examples.boolean_kernels import boolean_memory
    from t2m.frontend import capture_ttir
    from t2m.native import execute_native
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(boolean_memory, {"x":"*i1", "out":"*i1", "n":"i32", "B":"constexpr", "FILL":"constexpr"}, {"B":8, "FILL":fill}))
    abi = TensorABI("boolean_memory", (TensorBuffer("x", (17,), dtype="i1"), TensorBuffer("out", (17,), "readwrite", "i1")), grid=(3,),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0,17)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    assert any(t["kind"] == "logical-boolean-byte-memory" for t in result.trace["transformations"])
    for n in (0, 1, 13, 17):
        lanes = np.arange(17)
        x, initial = lanes % 2 == 0, lanes % 5 == 1
        expected = initial.copy()
        active = (lanes < n) & (lanes % 4 != 2)
        expected[active] = np.where(lanes % 3 != 1, x, fill)[active]
        outputs, _ = execute_native(result.mlir_text, [x, initial, np.asarray(n, np.int32)], tmp_path / f"native-{n}", python=os.environ["T2M_NATIVE_PYTHON"])
        np.testing.assert_array_equal(outputs[0], expected)
