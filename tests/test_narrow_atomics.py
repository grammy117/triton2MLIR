"""Source-width atomic words survive full public Merlin compilation/replay."""

import os
import signal
from pathlib import Path

import numpy as np
import pytest
from test_atomic_memory import operation_reference

from t2m.bundle import _record
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPS = ("add", "max", "umax", "min", "umin", "and", "or", "xor", "exch")
REQUIRED = all(os.environ.get(key) for key in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")


def program(mode):
    bits = 8 if mode == "i8" else 16
    dtype = f"i{bits}"
    groups = [("state", dtype, "integer")] if mode != "views" else [
        ("half", "f16", "word"), ("bytes", "i8", "bytes")]
    buffers = [TensorBuffer(name, (17 if kind == "bytes" else 8,), "readwrite", storage)
               for name, storage, kind in groups]
    buffers += [TensorBuffer("values", ("n",), dtype=dtype), TensorBuffer("indices", ("n",), dtype="i32")]
    for name, _, _ in groups:
        buffers += [TensorBuffer(f"old_{name}_{i}", ("n",), "write", dtype) for i in range(len(OPS))]
        buffers += [TensorBuffer(f"cas_{name}_{kind}", (1,), "write", dtype) for kind in ("ok", "no")]
    scalars = [RuntimeScalar("n", "i32", len(buffers), bounds=(0, 8))]
    if mode == "views":
        scalars.append(RuntimeScalar("origin", "i64", len(buffers) + 1, bounds=(0, 1)))
    abi = TensorABI("words", tuple(buffers), runtime_scalars=tuple(scalars))
    args = [f"%{buffer.name}:!tt.ptr<{buffer.dtype}>" for buffer in buffers]
    args += [f"%{scalar.name}:{scalar.dtype}" for scalar in scalars]
    source = ["module {tt.func public @words(" + ",".join(args) + ") {", f'''
      %r=tt.make_range {{start=0:i32,end=8:i32}}:tensor<8xi32>
      %nv=tt.splat %n:i32->tensor<8xi32>
      %live=arith.cmpi slt,%r,%nv:tensor<8xi32>
      %three=arith.constant dense<3>:tensor<8xi32>
      %one=arith.constant dense<1>:tensor<8xi32>
      %rem=arith.remsi %r,%three:tensor<8xi32>
      %sel=arith.cmpi ne,%rem,%one:tensor<8xi32>
      %active=arith.andi %live,%sel:tensor<8xi1>
      %ip=tt.splat %indices:!tt.ptr<i32>->tensor<8x!tt.ptr<i32>>
      %ia=tt.addptr %ip,%r:tensor<8x!tt.ptr<i32>>,tensor<8xi32>
      %iz=arith.constant dense<0>:tensor<8xi32>
      %index=tt.load %ia,%live,%iz:tensor<8x!tt.ptr<i32>>
      %vp=tt.splat %values:!tt.ptr<{dtype}>->tensor<8x!tt.ptr<{dtype}>>
      %va=tt.addptr %vp,%r:tensor<8x!tt.ptr<{dtype}>>,tensor<8xi32>
      %vz=arith.constant dense<0>:tensor<8x{dtype}>
      %value=tt.load %va,%live,%vz:tensor<8x!tt.ptr<{dtype}>>
      %sum="tt.reduce"(%value) ({{
        ^bb0(%a:{dtype},%b:{dtype}):
          %s=arith.addi %a,%b:{dtype}
          tt.reduce.return %s:{dtype}
      }}) {{axis=0:i32}}:(tensor<8x{dtype}>)->{dtype}
      %zero=arith.constant 0:{dtype}
      %bit=arith.constant 1:{dtype}
      %zero32=arith.constant 0:i32
      %run=arith.cmpi sgt,%n,%zero32:i32
    ''']
    for name, storage, kind in groups:
        if kind == "bytes":
            source.append(f'''
              %iv64_{name}=arith.extsi %index:tensor<8xi32> to tensor<8xi64>
              %two_{name}=arith.constant dense<2>:tensor<8xi64>
              %scaled_{name}=arith.muli %iv64_{name},%two_{name}:tensor<8xi64>
              %origin_{name}=tt.splat %origin:i64->tensor<8xi64>
              %offset_{name}=arith.addi %scaled_{name},%origin_{name}:tensor<8xi64>
              %bp_{name}=tt.splat %{name}:!tt.ptr<i8>->tensor<8x!tt.ptr<i8>>
              %ba_{name}=tt.addptr %bp_{name},%offset_{name}:tensor<8x!tt.ptr<i8>>,tensor<8xi64>
              %ptr_{name}=tt.bitcast %ba_{name}:tensor<8x!tt.ptr<i8>>->tensor<8x!tt.ptr<{dtype}>>
              %b0_{name}=tt.addptr %{name},%origin:!tt.ptr<i8>,i64
              %scalar_{name}=tt.bitcast %b0_{name}:!tt.ptr<i8>->!tt.ptr<{dtype}>
            ''')
        else:
            scalar = name
            if kind == "word":
                source.append(f"%scalar_{name}=tt.bitcast %{name}:!tt.ptr<{storage}>->!tt.ptr<{dtype}>")
                scalar = "scalar_" + name
            source.append(f'''
              %sp_{name}=tt.splat %{scalar}:!tt.ptr<{dtype}>->tensor<8x!tt.ptr<{dtype}>>
              %ptr_{name}=tt.addptr %sp_{name},%index:tensor<8x!tt.ptr<{dtype}>>,tensor<8xi32>
            ''')
        scalar = name if kind == "integer" else "scalar_" + name
        for i, operation in enumerate(OPS):
            source.append(f'''
              %result_{name}_{i}=tt.atomic_rmw {operation}, relaxed, gpu, %ptr_{name}, %value, %active:
                (tensor<8x!tt.ptr<{dtype}>>,tensor<8x{dtype}>,tensor<8xi1>)->tensor<8x{dtype}>
              %op_{name}_{i}=tt.splat %old_{name}_{i}:!tt.ptr<{dtype}>->tensor<8x!tt.ptr<{dtype}>>
            ''')
            source.append(f'''
              %oa_{name}_{i}=tt.addptr %op_{name}_{i},%r:tensor<8x!tt.ptr<{dtype}>>,tensor<8xi32>
              tt.store %oa_{name}_{i},%result_{name}_{i},%live:tensor<8x!tt.ptr<{dtype}>>
            ''')
        source.append(f'''
          tt.store %cas_{name}_ok,%zero:!tt.ptr<{dtype}>
          tt.store %cas_{name}_no,%zero:!tt.ptr<{dtype}>
          scf.if %run {{
            %current_{name}=tt.load %{scalar}:!tt.ptr<{dtype}>
            %before_{name}=tt.atomic_cas acq_rel, gpu, %{scalar}, %current_{name}, %sum:
              (!tt.ptr<{dtype}>,{dtype},{dtype})->{dtype}
            tt.store %cas_{name}_ok,%before_{name}:!tt.ptr<{dtype}>
            %different_{name}=arith.xori %sum,%bit:{dtype}
            %failed_{name}=tt.atomic_cas acquire, gpu, %{scalar}, %different_{name}, %zero:
              (!tt.ptr<{dtype}>,{dtype},{dtype})->{dtype}
            tt.store %cas_{name}_no,%failed_{name}:!tt.ptr<{dtype}>
          }}
        ''')
    source.append("tt.return }}")
    return "\n".join(source), abi, groups, bits


def word(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


def arguments(mode, n, change):
    _, _, groups, bits = program(mode)
    dtype = getattr(np, f"int{bits}")
    maximum = (1 << (bits - 1)) - 1
    initial = np.array([maximum, -maximum - 1, -1, 0, 7, -5, 3, -19], dtype=dtype)
    values = np.resize(np.array([1, -1, maximum, -maximum - 1, 5, 7, -3, 11], dtype=dtype), n)
    if change:
        initial = np.array([word(int(v) + change, bits) for v in initial], dtype=dtype)
        values = np.roll(values, change).copy()
    indices = ((np.arange(n, dtype=np.int32) * (5 if change else 3)) % 8).copy()
    active = np.arange(n) % 3 != 1
    indices[~active] = -1000
    inputs, states, snapshots = [], [], []
    for _, _, kind in groups:
        state = initial.copy()
        old = [np.zeros(n, dtype) for _ in OPS]
        for i, operation in enumerate(OPS):
            for lane in range(n):
                if not active[lane]:
                    continue
                position = indices[lane]
                a, b = int(state[position]), int(values[lane])
                old[i][lane] = a
                state[position] = operation_reference(a, b, operation, bits)
        before, failure = np.zeros(1, dtype), np.zeros(1, dtype)
        if n:
            before[0] = state[0]
            state[0] = word(sum(int(v) for v in values), bits)
            failure[0] = state[0]
        if kind == "word":
            inputs.append(initial.view(np.float16).copy())
            states.append(state.view(np.float16).copy())
        elif kind == "bytes":
            inputs.append(np.concatenate((initial.view(np.int8), np.array([0x5A], np.int8))))
            states.append(np.concatenate((state.view(np.int8), np.array([0x5A], np.int8))))
        else:
            inputs.append(initial.copy())
            states.append(state)
        snapshots.extend([*old, before, failure])
    inputs.extend([values, indices, np.asarray(n, np.int32)])
    if mode == "views":
        inputs.append(np.asarray(0, np.int64))
    return inputs, [*states, *snapshots]


def invoke(root, arrays, name, *, error=None):
    from t2m import consumer
    from t2m.process_outcome import observe

    inputs, out = root / (name + ".npz"), root / name
    np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
    out.mkdir()
    command = [os.environ["T2M_MERLIN_PYTHON"], str(Path(consumer.__file__).with_name("consumer_worker.py")),
        "observe", os.environ["T2M_MERLIN_ROOT"], str(root / "compiled/frontend"), str(root / "compiled/host"), str(inputs), str(out)]
    outcome = observe(command, out, environment=dict(os.environ, TMPDIR=str(out)), timeout=60)
    assert not outcome["timed_out"] and (out / "invocation-started.json").is_file()
    if error:
        assert outcome["returncode"] == -signal.SIGABRT
        assert error.encode() in (out / "stdout.bin").read_bytes() + (out / "stderr.bin").read_bytes()
        assert not (out / "outputs.npz").exists()
        return None
    assert outcome["returncode"] == 0, (out / "stderr.bin").read_text()
    with np.load(out / "outputs.npz", allow_pickle=False) as outputs:
        assert outputs.files == [f"out{i}" for i in range(len(outputs.files))]
        return [outputs[name].copy() for name in outputs.files]


@pytest.mark.parametrize("mode", ["i8", "i16", "views"])
def test_narrow_atomics_preserve_all_old_final_words_and_guards(tmp_path, mode):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    source, abi, groups, bits = program(mode)
    (tmp_path / "source.ttir").write_text(source)
    result = convert_shared(tmp_path / "source.ttir", abi, tmp_path / "translation", memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="atomics" if mode == "i8" else "typed-atomics",
        byte_order="little" if mode == "views" else None)
    assert sum(row["opcode"] == "tt.atomic_rmw" for row in result.trace["source_operations"]) == len(groups) * len(OPS)
    assert all(row["word_bits"] == bits for row in result.trace["memory"]["atomic_payload"])
    assert not any(op.name.startswith(("tt.", "tts.", "gpu.", "llvm.")) for op in result.module.walk())
    arrays, expected = arguments(mode, 8, 0)
    write_shared_bundle(result, tmp_path / "capture", arrays, expected, work=tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    stage_compile_inputs(tmp_path / "capture", tmp_path / "stage")
    product = compile(StagedProgram(tmp_path / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=tmp_path / "compiled")
    product.verify()
    library = _record(tmp_path / "compiled/host/model.so")
    assert not any(p.name in {"inputs.npz", "golden.npy", "extra.npz"}
                   for p in (tmp_path / "compiled/frontend").rglob("*"))
    for number, (n, change) in enumerate(((8, 0), (5, 3), (1, 0), (0, 0))):
        arrays, expected = arguments(mode, n, change)
        before = [value.tobytes() for value in arrays]
        actual = invoke(tmp_path, arrays, "invoke-" + str(number))
        assert len(actual) == len(expected) == 12 * len(groups)
        for value, reference in zip(actual, expected, strict=True):
            assert (value.shape, value.dtype, value.tobytes()) == (reference.shape, reference.dtype, reference.tobytes())
        assert [value.tobytes() for value in arrays] == before
    for number, invalid in enumerate((-1, 8)):
        arrays, _ = arguments(mode, 1, 0)
        arrays[len(groups) + 1][0] = invalid
        invoke(tmp_path, arrays, "bad-index-" + str(number), error="Triton active memory index outside logical allocation")
    if mode == "views":
        arrays, _ = arguments(mode, 1, 0)
        arrays[-1] = np.asarray(1, np.int64)
        invoke(tmp_path, arrays, "bad-alignment", error="Triton byte atomic address is not word aligned")
        arrays, expected = arguments(mode, 0, 0)
        arrays[-1] = np.asarray(1, np.int64)
        actual = invoke(tmp_path, arrays, "inactive-alignment")
        assert [a.tobytes() for a in actual] == [a.tobytes() for a in expected]
    assert _record(tmp_path / "compiled/host/model.so") == library
    product.verify()
