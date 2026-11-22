"""Exact low-precision values/words and streams through public Merlin."""

import json
import math
import os

import numpy as np
import pytest
from test_narrow_atomics import REQUIRED, invoke

from t2m.bundle import _record
from t2m.effect_bundle import write_effect_bundle
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
from t2m.tensor_storage import FP8_NUMPY_NAMES

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")


def arguments(dtype, n, steps, shift):
    import ml_dtypes

    words = ((np.arange(n, dtype=np.uint16) + shift) % 256).astype(np.uint8)
    integers = ((np.arange(n, dtype=np.uint16) * 7 + shift) % 256).astype(np.uint8).view(np.int8)
    decoded = words.view(getattr(ml_dtypes, FP8_NUMPY_NAMES[dtype])).astype(np.float64)
    small = np.asarray([((int(w) + 8) % 16) - 8 for w in integers], np.int8)
    lines = []
    for i, (word, value) in enumerate(zip(words, decoded, strict=True)):
        if math.isnan(value):
            printed = "nan" if dtype.endswith("FNUZ") or int(word) < 128 else "-nan"
        else:
            printed = f"{value:.6f}"
        for phase in range(steps):
            if phase % 2:
                continue
            values = (("fp", printed), ("fphex", f"0x{word:02x}"),
                      ("signed4", str(small[i])), ("unsigned4", str(int(integers[i]) & 15)),
                      ("hex4", f"0x{int(integers[i]) & 15:x}"))
            lines.extend(f"pid (0, 0, 0) idx () {label}: {text}\n" for label, text in values)
    arrays = [words, integers, np.asarray(n, np.int32), np.asarray(steps, np.int32)]
    return arrays, [words.copy(), small], "".join(lines)


@pytest.mark.parametrize("dtype", FP8_NUMPY_NAMES)
def test_all_fp8_words_and_internal_i4_keep_exact_streams_and_outputs(tmp_path, dtype):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    from t2m.merlin_backend import MerlinHostBackend

    source = f'''module {{tt.func public @show_low(
      %x:!tt.ptr<{dtype}>,%words:!tt.ptr<i8>,%out:!tt.ptr<{dtype}>,%small_out:!tt.ptr<i8>,%n:i32,%steps:i32) {{
      %zero=arith.constant 0:i32
      %one=arith.constant 1:i32
      %two=arith.constant 2:i32
      scf.for %i=%zero to %n step %one : i32 {{
        %xp=tt.addptr %x,%i:!tt.ptr<{dtype}>,i32
        %ip=tt.addptr %words,%i:!tt.ptr<i8>,i32
        %v=tt.load %xp:!tt.ptr<{dtype}>
        %w=tt.load %ip:!tt.ptr<i8>
        %small=arith.trunci %w:i8 to i4
        %signed=arith.extsi %small:i4 to i8
        %op=tt.addptr %out,%i:!tt.ptr<{dtype}>,i32
        %sp=tt.addptr %small_out,%i:!tt.ptr<i8>,i32
        tt.store %op,%v:!tt.ptr<{dtype}>
        tt.store %sp,%signed:!tt.ptr<i8>
        scf.for %phase=%zero to %steps step %one : i32 {{
          %parity=arith.remui %phase,%two:i32
          %enabled=arith.cmpi eq,%parity,%zero:i32
          scf.if %enabled {{
            tt.print " fp: " {{hex=false,isSigned=array<i32:1>}} : %v:{dtype}
            tt.print " fphex: " {{hex=true,isSigned=array<i32:0>}} : %v:{dtype}
            tt.print " signed4: " {{hex=false,isSigned=array<i32:1>}} : %small:i4
            tt.print " unsigned4: " {{hex=false,isSigned=array<i32:0>}} : %small:i4
            tt.print " hex4: " {{hex=true,isSigned=array<i32:0>}} : %small:i4
          }}
        }}
      }}
      tt.return
    }}}}'''
    abi = TensorABI("show_low", (
        TensorBuffer("x", ("n",), dtype=dtype), TensorBuffer("words", ("n",), dtype="i8"),
        TensorBuffer("out", ("n",), "write", dtype), TensorBuffer("small_out", ("n",), "write", "i8")),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 256)),
                         RuntimeScalar("steps", "i32", 5, bounds=(0, 3))))
    (tmp_path / "source.ttir").write_text(source)
    result = convert_shared(tmp_path / "source.ttir", abi, tmp_path / "translation",
        memory_profile="guarded-buffer", address_policy="source-words")
    assert not any(op.name.startswith(("tt.", "tts.", "gpu.", "llvm.")) for op in result.module.walk())
    arrays, expected, stdout = arguments(dtype, 256, 1, 0)
    reference = {"schema": "t2m.effect-reference.v2", "termination": "return", "stdout": stdout,
        "stderr": "", "results": {"archive": "reference.npz", "comparisons": [{"kind": "words"}] * 2}}
    write_effect_bundle(result, tmp_path / "capture", arrays, reference, references=expected,
        work=tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    stage_compile_inputs(tmp_path / "capture", tmp_path / "stage")
    product = compile(StagedProgram(tmp_path / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=tmp_path / "compiled")
    product.verify()
    features = json.loads((tmp_path / "compiled/host/compilation.json").read_text())["features"]
    assert "lower_fp8_widening" in features and "lower_scalar_diagnostics" in features
    library = _record(tmp_path / "compiled/host/model.so")
    assert not any(p.name in {"inputs.npz", "reference.json", "reference.npz", "extra.npz"}
        for p in (tmp_path / "compiled/frontend").rglob("*"))
    for number, (n, steps, shift) in enumerate(((256, 1, 0), (256, 1, 113), (17, 3, 129), (1, 0, 255), (0, 3, 0))):
        arrays, expected, stdout = arguments(dtype, n, steps, shift)
        before = [value.tobytes() for value in arrays]
        name = f"invoke-{number}"
        actual = invoke(tmp_path, arrays, name)
        assert len(actual) == len(expected) == 2
        for value, reference in zip(actual, expected, strict=True):
            assert (value.shape, value.dtype, value.tobytes()) == (reference.shape, reference.dtype, reference.tobytes())
        assert (tmp_path / name / "stdout.bin").read_bytes() == stdout.encode()
        assert (tmp_path / name / "stderr.bin").read_bytes() == b""
        assert [value.tobytes() for value in arrays] == before
    assert _record(tmp_path / "compiled/host/model.so") == library
    product.verify()
