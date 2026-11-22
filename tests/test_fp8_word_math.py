"""Word-preserving FP8 negation, absolute value and copy-sign."""

import os

import numpy as np
import pytest
from test_narrow_atomics import REQUIRED, invoke

from t2m.bundle import _record
from t2m.dynamic_abi import GridExtent
from t2m.frontend import inspect_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
from t2m.tensor_storage import FP8_NUMPY_NAMES

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")


def reference(arrays, dtype):
    import ml_dtypes

    type_ = getattr(ml_dtypes, FP8_NUMPY_NAMES[dtype])
    x, y = (array.view(type_) for array in arrays[:2])
    negative, absolute, signed = np.negative(x), np.abs(x), np.copysign(x, y)
    if dtype.endswith("FNUZ"):
        # APFloat::changeSign leaves unsigned zero and the unique NaN intact.
        # NumPy's copysign loop instead copies raw sign bits, which can turn
        # zero into NaN or vice versa. Apply the pinned format's semantic rule.
        special = np.isnan(x) | (x == 0)
        signed = np.where(special, x, signed).astype(type_)
    return [array.view(np.uint8) for array in (negative, absolute, signed)]


@pytest.mark.parametrize("dtype", tuple(FP8_NUMPY_NAMES))
def test_every_word_and_sign_source_preserves_exact_fp8_encodings(tmp_path, dtype):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    names = ("x", "y", "negative", "absolute", "signed")
    abi = TensorABI("words", tuple(TensorBuffer(name, ("n",), "read" if i < 2 else "write", dtype)
        for i, name in enumerate(names)), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 5, bounds=(0, 65536)),))
    typ, ptr = f"tensor<8x{dtype}>", f"tensor<8x!tt.ptr<{dtype}>>"
    args = ",".join(f"%{name}:!tt.ptr<{dtype}>" for name in names)
    lines = [f"module {{tt.func public @words({args},%n:i32) {{", '''
      %pid=tt.get_program_id x:i32
      %eight=arith.constant 8:i32
      %base=arith.muli %pid,%eight:i32
      %r=tt.make_range {start=0:i32,end=8:i32}:tensor<8xi32>
      %bases=tt.splat %base:i32->tensor<8xi32>
      %lane=arith.addi %r,%bases:tensor<8xi32>
      %ns=tt.splat %n:i32->tensor<8xi32>
      %live=arith.cmpi slt,%lane,%ns:tensor<8xi32>
    ''', f"%zero=arith.constant dense<0.0>:{typ}"]
    for name in names:
        lines += [f"%p_{name}=tt.splat %{name}:!tt.ptr<{dtype}>->{ptr}",
            f"%q_{name}=tt.addptr %p_{name},%lane:{ptr},tensor<8xi32>"]
    for name in names[:2]:
        lines.append(f"%v_{name}=tt.load %q_{name},%live,%zero:{ptr}")
    lines += [f"%neg=arith.negf %v_x:{typ}", f"%abs=math.absf %v_x:{typ}", f"%sign=math.copysign %v_x,%v_y:{typ}",
        f"tt.store %q_negative,%neg,%live:{ptr}", f"tt.store %q_absolute,%abs,%live:{ptr}",
        f"tt.store %q_signed,%sign,%live:{ptr}", "tt.return", "}}"]
    path = tmp_path / "source.ttir"
    path.write_text("\n".join(lines))
    inspect_ttir(path, tmp_path / "verification")
    result = convert_shared(path, abi, tmp_path / "translation", memory_profile="guarded-buffer", address_policy="source-words")

    def arguments(n, shift):
        indices = np.arange(n, dtype=np.uint32)
        return [((indices // 256 + shift) & 255).astype(np.uint8),
            ((indices + 7 * shift) & 255).astype(np.uint8), np.asarray(n, np.int32)]

    arrays = arguments(65536, 0)
    write_shared_bundle(result, tmp_path / "capture", arrays, reference(arrays, dtype), work=tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"])
    stage_compile_inputs(tmp_path / "capture", tmp_path / "stage")
    product = compile(StagedProgram(tmp_path / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=tmp_path / "compiled")
    library = _record(tmp_path / "compiled/host/model.so")
    for index, (n, shift) in enumerate(((65536, 0), (65536, 31), (257, 127), (1, 128), (0, 0))):
        arrays = arguments(n, shift)
        before = [a.tobytes() for a in arrays]
        actual = invoke(tmp_path, arrays, f"invoke-{index}")
        assert len(actual) == 3
        for got, want in zip(actual, reference(arrays, dtype), strict=True):
            assert (got.shape, got.dtype, got.tobytes()) == (want.shape, np.uint8, want.tobytes())
        assert before == [a.tobytes() for a in arrays]
    assert _record(tmp_path / "compiled/host/model.so") == library
    product.verify()
