"""Closed address computations use actual unpacked public cell footprints."""

import os

import numpy as np
import pytest
from test_narrow_atomics import REQUIRED, invoke

from t2m.bundle import _record
from t2m.frontend import inspect_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")


@pytest.mark.parametrize("dtype", ["i1", "i4"])
def test_unpacked_cells_keep_closed_roundtrips_byte_steps_and_differences(tmp_path, dtype):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    abi = TensorABI("cells", (
        TensorBuffer("x", (8,), dtype=dtype), TensorBuffer("out", (2,), "write", dtype),
        TensorBuffer("distance", (1,), "write", "i64")),
        runtime_scalars=(RuntimeScalar("index", "i32", 3, bounds=(-1, 8)),))
    source = f'''module {{tt.func public @cells(%x:!tt.ptr<{dtype}>,%out:!tt.ptr<{dtype}>,
        %distance:!tt.ptr<i64>,%index:i32) {{
        %p=tt.addptr %x,%index:!tt.ptr<{dtype}>,i32
        %base=tt.ptr_to_int %x:!tt.ptr<{dtype}>->i64
        %address=tt.ptr_to_int %p:!tt.ptr<{dtype}>->i64
        %delta=arith.subi %address,%base:i64
        tt.store %distance,%delta:!tt.ptr<i64>
        %back=tt.int_to_ptr %address:i64->!tt.ptr<{dtype}>
        %value=tt.load %back:!tt.ptr<{dtype}>
        tt.store %out,%value:!tt.ptr<{dtype}>
        %one=arith.constant 1:i64
        %next=arith.addi %address,%one:i64
        %nextptr=tt.int_to_ptr %next:i64->!tt.ptr<{dtype}>
        %nextvalue=tt.load %nextptr:!tt.ptr<{dtype}>
        %second=tt.addptr %out,%one:!tt.ptr<{dtype}>,i64
        tt.store %second,%nextvalue:!tt.ptr<{dtype}>
        tt.return
    }}}}'''
    path = tmp_path / "source.ttir"
    path.write_text(source)
    inspect_ttir(path, tmp_path / "original-verification")
    result = convert_shared(path, abi, tmp_path / "translation",
        memory_profile="guarded-buffer", address_policy="source-words")
    assert any(row.get("kind") == "closed-pointer-integer-difference" for row in result.trace["transformations"])
    assert not any(op.name in ("tt.ptr_to_int", "tt.int_to_ptr") for op in result.module.walk())

    def arguments(index, shift):
        words = (np.arange(8, dtype=np.uint8) * 3 + shift) & (1 if dtype == "i1" else 15)
        values = words.astype(np.bool_) if dtype == "i1" else words
        return [values, np.asarray(index, np.int32)]

    arrays = arguments(0, 0)
    expected = [arrays[0][:2].copy(), np.asarray([0], np.int64)]
    write_shared_bundle(result, tmp_path / "capture", arrays, expected,
        work=tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    stage_compile_inputs(tmp_path / "capture", tmp_path / "stage")
    product = compile(StagedProgram(tmp_path / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]),
        output=tmp_path / "compiled")
    library = _record(tmp_path / "compiled/host/model.so")
    for index, shift in ((0, 0), (3, 5), (6, 11)):
        arrays = arguments(index, shift)
        original = arrays[0].tobytes()
        output = invoke(tmp_path, arrays, f"index-{index}")
        assert len(output) == 2
        assert (output[0].dtype, output[0].tobytes()) == (arrays[0].dtype, arrays[0][index:index + 2].tobytes())
        assert (output[1].dtype, output[1].tolist()) == (np.int64, [index])
        assert arrays[0].tobytes() == original
    for invalid in (-1, 7, 8):
        invoke(tmp_path, arguments(invalid, 0), f"bad-{invalid}",
            error="Triton active memory index outside logical allocation")
    assert _record(tmp_path / "compiled/host/model.so") == library
    product.verify()
