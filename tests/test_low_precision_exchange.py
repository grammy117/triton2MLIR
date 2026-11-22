"""Bitwise low-precision exchange keeps all words through public Merlin replay."""

import os

import numpy as np
import pytest
from test_narrow_atomics import invoke

from t2m.bundle import _record
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
from t2m.tensor_storage import DTYPES

FORMATS = ("f16", "bf16", "f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ")
REQUIRED = all(os.environ.get(key) for key in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")


def program(dtype):
    bits = 16 if dtype in ("f16", "bf16") else 8
    capacity = 1 << bits
    abi = TensorABI("exchange", (
        TensorBuffer("state", (capacity,), "readwrite", dtype),
        TensorBuffer("scalar", (1,), "readwrite", dtype),
        TensorBuffer("values", ("n",), dtype=dtype),
        TensorBuffer("indices", ("n",), dtype="i32"),
        TensorBuffer("old", ("n",), "write", dtype),
        TensorBuffer("scalar_old", ("n",), "write", dtype)),
        runtime_scalars=(RuntimeScalar("n", "i32", 6, bounds=(0, capacity)),
                         RuntimeScalar("phase", "i32", 7, bounds=(0, 2))))
    args = [f"%{b.name}:!tt.ptr<{b.dtype}>" for b in abi.buffers] + ["%n:i32", "%phase:i32"]
    source = "module {tt.func public @exchange(" + ",".join(args) + ") {" + f'''
      %range=tt.make_range {{start=0:i32,end=8:i32}}:tensor<8xi32>
      %zero=arith.constant 0:i32
      %eight=arith.constant 8:i32
      %three=arith.constant dense<3>:tensor<8xi32>
      %one=arith.constant dense<1>:tensor<8xi32>
      %fz=arith.constant 0.0:{dtype}
      %fill=tt.splat %fz:{dtype}->tensor<8x{dtype}>
      %nv=tt.splat %n:i32->tensor<8xi32>
      %phasev=tt.splat %phase:i32->tensor<8xi32>
      scf.for %i=%zero to %n step %eight : i32 {{
        %base=tt.splat %i:i32->tensor<8xi32>
        %r=arith.addi %base,%range:tensor<8xi32>
        %live=arith.cmpi slt,%r,%nv:tensor<8xi32>
        %shifted=arith.addi %r,%phasev:tensor<8xi32>
        %rem=arith.remsi %shifted,%three:tensor<8xi32>
        %selected=arith.cmpi ne,%rem,%one:tensor<8xi32>
        %active=arith.andi %live,%selected:tensor<8xi1>
        %vp=tt.splat %values:!tt.ptr<{dtype}>->tensor<8x!tt.ptr<{dtype}>>
        %va=tt.addptr %vp,%r:tensor<8x!tt.ptr<{dtype}>>,tensor<8xi32>
        %value=tt.load %va,%live,%fill:tensor<8x!tt.ptr<{dtype}>>
        %ip=tt.splat %indices:!tt.ptr<i32>->tensor<8x!tt.ptr<i32>>
        %ia=tt.addptr %ip,%r:tensor<8x!tt.ptr<i32>>,tensor<8xi32>
        %iz=arith.constant dense<0>:tensor<8xi32>
        %index=tt.load %ia,%live,%iz:tensor<8x!tt.ptr<i32>>
        %sp=tt.splat %state:!tt.ptr<{dtype}>->tensor<8x!tt.ptr<{dtype}>>
        %sa=tt.addptr %sp,%index:tensor<8x!tt.ptr<{dtype}>>,tensor<8xi32>
        %before=tt.atomic_rmw exch, relaxed, gpu, %sa,%value,%active:
          (tensor<8x!tt.ptr<{dtype}>>,tensor<8x{dtype}>,tensor<8xi1>)->tensor<8x{dtype}>
        %op=tt.splat %old:!tt.ptr<{dtype}>->tensor<8x!tt.ptr<{dtype}>>
        %oa=tt.addptr %op,%r:tensor<8x!tt.ptr<{dtype}>>,tensor<8xi32>
        tt.store %oa,%before,%live:tensor<8x!tt.ptr<{dtype}>>
        %hp=tt.splat %scalar_old:!tt.ptr<{dtype}>->tensor<8x!tt.ptr<{dtype}>>
        %ha=tt.addptr %hp,%r:tensor<8x!tt.ptr<{dtype}>>,tensor<8xi32>
        tt.store %ha,%fill,%live:tensor<8x!tt.ptr<{dtype}>>
        %single=tt.addptr %values,%i:!tt.ptr<{dtype}>,i32
        %next=tt.load %single:!tt.ptr<{dtype}>
        %prior=tt.atomic_rmw exch, acq_rel, gpu, %scalar,%next:
          (!tt.ptr<{dtype}>,{dtype})->{dtype}
        %history=tt.addptr %scalar_old,%i:!tt.ptr<{dtype}>,i32
        tt.store %history,%prior:!tt.ptr<{dtype}>
      }}
      tt.return
    }}}}'''
    return source, abi, bits, capacity


def arguments(dtype, n, shift):
    _, _, bits, capacity = program(dtype)
    word = np.dtype(f"uint{bits}")
    state = np.arange(capacity, dtype=word)
    state = np.roll(state, 0 if n == capacity else shift).copy()
    scalar = np.array([(capacity - 1 - shift) % capacity], dtype=word)
    values = ((np.arange(n, dtype=np.uint64) * 37 + (0 if n == capacity else shift)) % capacity).astype(word)
    indices = ((np.arange(n, dtype=np.int32) * 17) % capacity).copy()
    active = (np.arange(n) + shift % 3) % 3 != 1
    indices[~active] = -1000
    final, old = state.copy(), np.zeros(n, word)
    final[indices[active]] = values[active]
    old[active] = state[indices[active]]
    scalar_old, last = np.zeros(n, word), scalar.copy()
    for start in range(0, n, 8):
        scalar_old[start], last[0] = last[0], values[start]
    carrier = np.dtype(DTYPES[dtype])
    arrays = [state.view(carrier), scalar.view(carrier), values.view(carrier), indices,
              np.asarray(n, np.int32), np.asarray(shift % 3, np.int32)]
    expected = [v.view(carrier) for v in (final, last, old, scalar_old)]
    return arrays, expected


@pytest.mark.parametrize("dtype", FORMATS)
def test_every_low_precision_word_survives_scalar_and_tiled_exchange(tmp_path, dtype):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    source, abi, bits, capacity = program(dtype)
    path = tmp_path / "source.ttir"
    path.write_text(source)
    result = convert_shared(path, abi, tmp_path / "translation", memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    assert sum(row["opcode"] == "tt.atomic_rmw" for row in result.trace["source_operations"]) == 2
    assert all(row["word_bits"] == bits and row["numeric_contract"] == "bitwise-exchange"
               and row["value_dtype"] == dtype for row in result.trace["memory"]["atomic_payload"])
    arrays, expected = arguments(dtype, 13, 0)
    write_shared_bundle(result, tmp_path / "capture", arrays, expected, work=tmp_path / "native",
                        python=os.environ["T2M_NATIVE_PYTHON"])
    stage_compile_inputs(tmp_path / "capture", tmp_path / "stage")
    product = compile(StagedProgram(tmp_path / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=tmp_path / "compiled")
    product.verify()
    library = _record(tmp_path / "compiled/host/model.so")
    assert not any(p.name in {"inputs.npz", "golden.npy", "extra.npz"}
                   for p in (tmp_path / "compiled/frontend").rglob("*"))
    seen_old, seen_new = np.zeros(capacity, np.bool_), np.zeros(capacity, np.bool_)
    for number, (n, shift) in enumerate(((capacity, 0), (capacity, 1), (13, 3), (1, 2), (0, 0))):
        arrays, expected = arguments(dtype, n, shift)
        before = [a.tobytes() for a in arrays]
        actual = invoke(tmp_path, arrays, "invoke-" + str(number))
        assert len(actual) == len(expected) == 4
        assert [(a.shape, a.dtype, a.tobytes()) for a in actual] == [(a.shape, a.dtype, a.tobytes()) for a in expected]
        assert [a.tobytes() for a in arrays] == before
        if n == capacity:
            active = (np.arange(n) + shift % 3) % 3 != 1
            seen_old[actual[2].view(f"uint{bits}")[active]] = True
            seen_new[arrays[2].view(f"uint{bits}")[active]] = True
    assert seen_old.all() and seen_new.all()
    for number, bad in enumerate((-1, capacity)):
        arrays, _ = arguments(dtype, 1, 0)
        arrays[3][0] = bad
        invoke(tmp_path, arrays, "bad-index-" + str(number), error="Triton active memory index outside logical allocation")
    assert _record(tmp_path / "compiled/host/model.so") == library
    product.verify()
