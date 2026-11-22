"""Source contraction permission does not erase FP8 rounding boundaries."""

import os

import numpy as np
import pytest
from test_narrow_atomics import REQUIRED, invoke

from t2m.bundle import _record
from t2m.examples import fp8_arithmetic_reference as oracle
from t2m.frontend import inspect_ttir
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")


def test_source_contract_flags_keep_each_fp8_rounding_and_fused_result(tmp_path):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs
    from test_fp8_arithmetic import compare

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    dtype = "f8E5M2"
    names = ("x", "y", "sum", "diff", "product", "fused")
    abi = TensorABI("flags", tuple(TensorBuffer(name, (1,), "read" if i < 2 else "write", dtype)
        for i, name in enumerate(names)))
    args = ",".join(f"%{name}:!tt.ptr<{dtype}>" for name in names)
    source = f'''module {{tt.func public @flags({args}) {{
      %a=tt.load %x:!tt.ptr<{dtype}>
      %b=tt.load %y:!tt.ptr<{dtype}>
      %s=arith.addf %a,%b fastmath<contract>:{dtype}
      %d=arith.subf %s,%b fastmath<contract>:{dtype}
      %p=arith.mulf %a,%b fastmath<contract>:{dtype}
      %f=math.fma %a,%b,%a fastmath<contract>:{dtype}
      tt.store %sum,%s:!tt.ptr<{dtype}>
      tt.store %diff,%d:!tt.ptr<{dtype}>
      tt.store %product,%p:!tt.ptr<{dtype}>
      tt.store %fused,%f:!tt.ptr<{dtype}>
      tt.return
    }}}}'''
    path = tmp_path / "source.ttir"
    path.write_text(source)
    inspect_ttir(path, tmp_path / "verification")
    result = convert_shared(path, abi, tmp_path / "translation", memory_profile="guarded-buffer", address_policy="source-words")

    def arguments(a, b):
        first = oracle.binary(a, b, "add", dtype)
        outputs = [first, oracle.binary(first, b, "sub", dtype), oracle.binary(a, b, "mul", dtype),
            oracle.fused(a, b, a, dtype)]
        return [np.asarray([v], np.uint8) for v in (a, b)], [np.asarray([v], np.uint8) for v in outputs]

    arrays, expected = arguments(60, 123)
    assert expected[1].tolist() == [0]
    write_shared_bundle(result, tmp_path / "capture", arrays, expected, work=tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"])
    stage_compile_inputs(tmp_path / "capture", tmp_path / "stage")
    product = compile(StagedProgram(tmp_path / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=tmp_path / "compiled")
    library = _record(tmp_path / "compiled/host/model.so")
    for index, (a, b) in enumerate(((60, 123), (61, 62), (189, 62), (125, 60), (128, 0))):
        arrays, expected = arguments(a, b)
        outputs = invoke(tmp_path, arrays, f"invoke-{index}")
        assert len(outputs) == 4
        for got, want in zip(outputs, expected, strict=True):
            compare(got, want, dtype)
    assert _record(tmp_path / "compiled/host/model.so") == library
    product.verify()
