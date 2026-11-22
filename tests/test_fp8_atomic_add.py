"""Source-width FP8 atomic effects through reusable public Merlin libraries."""

import json
import math
import os
from dataclasses import replace

import numpy as np
import pytest
from test_low_precision_exchange import REQUIRED
from test_low_precision_exchange import program as exchange_program
from test_narrow_atomics import invoke

from t2m.bundle import _record
from t2m.examples.fp8_arithmetic_reference import binary, rounded
from t2m.examples.fp8_reference import FORMATS, decode
from t2m.frontend import inspect_ttir
from t2m.shared_tensor import convert_shared

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")
CAPACITY = 65536


def program(dtype):
    # Reuse the existing tiled, masked and ordered-scalar storage fixture.
    source, abi, _, _ = exchange_program(dtype)
    source = source.replace("atomic_rmw exch", "atomic_rmw fadd").replace(
        "fadd, acq_rel, gpu", "fadd, acq_rel, sys")
    buffers = tuple(replace(b, shape=(CAPACITY,)) if b.name == "state" else b for b in abi.buffers)
    scalars = (replace(abi.runtime_scalars[0], bounds=(0, CAPACITY)), abi.runtime_scalars[1])
    return source, replace(abi, buffers=buffers, runtime_scalars=scalars)


def arguments(dtype, n, phase):
    state = (np.arange(CAPACITY, dtype=np.uint32) // 256).astype(np.uint8)
    values = (np.arange(n, dtype=np.uint32) & 255).astype(np.uint8)
    indices = np.arange(n, dtype=np.int32)
    active = (np.arange(n) + phase) % 3 != 1
    indices[~active] = -8191
    scalar = np.asarray([125], np.uint8)  # includes E5M2 signaling NaN
    final, old = state.copy(), np.zeros(n, np.uint8)
    for i in np.flatnonzero(active):
        old[i] = state[i]
        final[i] = binary(int(state[i]), int(values[i]), "add", dtype)
    history, last = np.zeros(n, np.uint8), scalar.copy()
    for i in range(0, n, 8):
        history[i] = last[0]
        last[0] = binary(int(last[0]), int(values[i]), "add", dtype)
    inputs = [state, scalar, values, indices, np.asarray(n, np.int32), np.asarray(phase, np.int32)]
    return inputs, [final, last, old, history]


def compare(actual, expected, dtype, *, exact=False):
    assert actual.dtype == expected.dtype == np.uint8 and actual.shape == expected.shape
    for got, want in zip(actual.flat, expected.flat, strict=True):
        if exact or not math.isnan(decode(int(want), dtype)):
            assert got == want, (dtype, int(got), int(want))
        else:
            assert math.isnan(decode(int(got), dtype))
            if dtype == "f8E5M2":
                if int(want) & 2:
                    assert int(got) & 2  # every computed NaN is quiet
                else:
                    assert got == want  # unchanged signaling storage/old word


@pytest.mark.parametrize("dtype", tuple(FORMATS))
def test_all_fp8_atomic_pairs_keep_old_words_rounding_masks_and_replay(tmp_path, dtype):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    source, abi = program(dtype)
    path = tmp_path / "source.ttir"
    path.write_text(source)
    inspect_ttir(path, tmp_path / "source-verification")
    result = convert_shared(path, abi, tmp_path / "translation", memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    effects = result.trace["memory"]["atomic_payload"]
    assert len(effects) == 2
    assert all(e["word_bits"] == 8 and e["numeric_contract"] == f"ieee-{dtype}-rne-add-quiet-nans" for e in effects)
    assert {e["source_order"] for e in effects} == {"relaxed", "acq_rel"}
    assert {e["source_scope"] for e in effects} == {"gpu", "sys"}
    assert "memref.generic_atomic_rmw" in result.mlir_text and "llvm." not in result.mlir_text
    initial, expected = arguments(dtype, 13, 0)
    write_shared_bundle(result, tmp_path / "capture", initial, expected, work=tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"])
    stage_compile_inputs(tmp_path / "capture", tmp_path / "stage",
        result_mode="execution-owned" if dtype == "f8E4M3FNUZ" else None)
    product = compile(StagedProgram(tmp_path / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=tmp_path / "compiled")
    library = _record(tmp_path / "compiled/host/model.so")
    for i, (n, phase) in enumerate(((CAPACITY, 0), (CAPACITY, 1), (CAPACITY, 2),
            (263, 1), (257, 2), (1, 0), (0, 0))):
        arrays, expected = arguments(dtype, n, phase)
        before = [a.tobytes() for a in arrays]
        actual = invoke(tmp_path, arrays, f"invoke-{i}")
        assert len(actual) == 4
        for j, (got, want) in enumerate(zip(actual, expected, strict=True)):
            compare(got, want, dtype, exact=j == 2)
        written = np.zeros(CAPACITY, np.bool_)
        written[arrays[3][arrays[3] >= 0]] = True
        np.testing.assert_array_equal(actual[0][~written], arrays[0][~written])
        assert before == [a.tobytes() for a in arrays]
    # Colliding effects may execute in any valid serialization. Inspect the
    # complete old-value multiset for each destination rather than lane order.
    one = rounded(1, dtype)
    arrays, _ = arguments(dtype, 13, 0)
    arrays[0].fill(0)
    arrays[1].fill(0)
    arrays[2].fill(one)
    active = np.arange(13) % 3 != 1
    arrays[3][active] = np.arange(13, dtype=np.int32)[active] % 2
    actual = invoke(tmp_path, arrays, "collisions")
    for target in (0, 1):
        selected = active & (arrays[3] == target)
        count = int(np.count_nonzero(selected))
        assert actual[0][target] == rounded(count, dtype)
        np.testing.assert_array_equal(np.sort(actual[2][selected]),
            np.sort(np.asarray([rounded(j, dtype) for j in range(count)], np.uint8)))
    assert np.all(actual[2][~active] == 0)
    bad, _ = arguments(dtype, 1, 0)
    bad[3][0] = CAPACITY
    invoke(tmp_path, bad, "invalid-index", error="Triton active memory index outside logical allocation")
    assert _record(tmp_path / "compiled/host/model.so") == library
    assert not any(p.name in {"inputs.npz", "golden.npy", "extra.npz"}
        for p in (tmp_path / "compiled/frontend").rglob("*"))
    compilation = json.loads((tmp_path / "compiled/host/compilation.json").read_text())
    assert "lower_atomic_word_views" in compilation["features"]
    product.verify()


@pytest.mark.parametrize("dtype", tuple(FORMATS))
def test_constant_zero_atomic_updates_preserve_old_signaling_words(tmp_path, dtype):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.native import execute_native
    from t2m.shared_bundle import write_shared_bundle
    from t2m.shared_tensor import TensorABI, TensorBuffer

    source = f'''module {{tt.func public @zeros(%state:!tt.ptr<{dtype}>,
        %first:!tt.ptr<{dtype}>,%second:!tt.ptr<{dtype}>) {{
      %z=arith.constant 0:i32
      %end=arith.constant 256:i32
      %one=arith.constant 1:i32
      %positive=arith.constant 0.0:{dtype}
      %negative=arith.constant -0.0:{dtype}
      scf.for %i=%z to %end step %one:i32 {{
        %p=tt.addptr %state,%i:!tt.ptr<{dtype}>,i32
        %a=tt.atomic_rmw fadd,relaxed,cta,%p,%positive:(!tt.ptr<{dtype}>,{dtype})->{dtype}
        %b=tt.atomic_rmw fadd,acq_rel,sys,%p,%negative:(!tt.ptr<{dtype}>,{dtype})->{dtype}
        %x=tt.addptr %first,%i:!tt.ptr<{dtype}>,i32
        %y=tt.addptr %second,%i:!tt.ptr<{dtype}>,i32
        tt.store %x,%a:!tt.ptr<{dtype}>
        tt.store %y,%b:!tt.ptr<{dtype}>
      }}
      tt.return
    }}}}'''
    abi = TensorABI("zeros", (TensorBuffer("state", (256,), "readwrite", dtype),
        TensorBuffer("first", (256,), "write", dtype), TensorBuffer("second", (256,), "write", dtype)))
    path = tmp_path / "source.ttir"
    path.write_text(source)
    inspect_ttir(path, tmp_path / "source-verification")
    result = convert_shared(path, abi, tmp_path / "translation", memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    words = np.arange(256, dtype=np.uint8)
    actual, _ = execute_native(result.mlir_text, [words], tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
    positive = np.asarray([binary(int(v), 0, "add", dtype) for v in words], np.uint8)
    negative_zero = 0 if dtype.endswith("FNUZ") else 128
    final = np.asarray([binary(int(v), negative_zero, "add", dtype) for v in positive], np.uint8)
    compare(actual[1], words, dtype, exact=True)
    compare(actual[2], positive, dtype)
    compare(actual[0], final, dtype)
    write_shared_bundle(result, tmp_path / "capture", [words], [final, words, positive],
        work=tmp_path / "bundle-native", python=os.environ["T2M_NATIVE_PYTHON"])
    stage_compile_inputs(tmp_path / "capture", tmp_path / "stage")
    product = compile(StagedProgram(tmp_path / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=tmp_path / "compiled")
    actual = invoke(tmp_path, [words], "invoke-zeros")
    compare(actual[1], words, dtype, exact=True)
    compare(actual[2], positive, dtype)
    compare(actual[0], final, dtype)
    product.verify()


@pytest.mark.parametrize("dtype", tuple(FORMATS))
def test_native_fp8_atomic_borrowing_checks_complete_helper(tmp_path, dtype):
    from pathlib import Path

    from t2m.process_outcome import observe

    worker = tmp_path / "check.py"
    worker.write_text('''import sys
sys.path.insert(0,sys.argv[1]+"/src")
from torch_mlir import ir
from torch_mlir.passmanager import PassManager
from merlin.compiler.lower.options import atomic_compare_exchange as atomic
scope={"ir":ir}
exec(atomic.SUPPORT,scope)
with ir.Context() as ctx:
    for identity,wide in ((False,False),(False,True),(True,False)):
        source=scope["_fp8_add_source"]("checked",sys.argv[2],identity,wide)
        mutations=("none","address","retry","return")+(() if identity else ("arithmetic",))
        for mutation in mutations:
            module=ir.Module.parse(source)
            PassManager.parse("builtin.module(canonicalize,cse)").run(module.operation)
            function=tuple(module.body.operations)[0].operation
            if mutation in ("address","arithmetic"):
                oldname,newname=("arith.addi","arith.subi") if mutation=="address" else ("arith.addf","arith.subf")
                operation=next(o for o in scope["_cas_walk"](function) if o.name==oldname)
                with ir.InsertionPoint(operation), operation.location:
                    replacement=ir.Operation.create(newname,results=[operation.results[0].type],operands=list(operation.operands))
                operation.results[0].replace_all_uses_with(replacement.results[0])
                operation.erase()
            elif mutation=="retry":
                operation=next(o for o in scope["_cas_walk"](function) if o.name=="scf.condition")
                condition=operation.operands[0].owner.operation
                operation.operands[0]=condition.operands[0]
            elif mutation=="return":
                operation=next(o for o in scope["_cas_walk"](function) if o.name=="func.return")
                with ir.InsertionPoint(operation), operation.location:
                    zero=ir.Operation.create("arith.constant",results=[ir.IntegerType.get_signless(8)],attributes={"value":ir.IntegerAttr.get(ir.IntegerType.get_signless(8),0)})
                operation.operands[0]=zero.results[0]
            module.operation.verify()
            accepted=scope["_word_helper_equal"](function,"checked",module)
            assert accepted==(mutation=="none"),(identity,wide,mutation)
            print(identity,wide,mutation,"checked",flush=True)
''')
    output = tmp_path / "check"
    output.mkdir()
    outcome = observe([os.environ["T2M_NATIVE_PYTHON"], str(Path(worker)),
        os.environ["T2M_MERLIN_ROOT"], dtype], output, environment=dict(os.environ, TMPDIR=str(output)), timeout=60)
    assert not outcome["timed_out"] and outcome["returncode"] == 0, (output / "stderr.bin").read_text()
    assert (output / "stdout.bin").read_text().count("checked") == 14
