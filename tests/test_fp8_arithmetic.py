"""FP8 rounding, reusable tiles and atomic exchange through public Merlin."""

import json
import math
import os

import numpy as np
import pytest
from test_narrow_atomics import REQUIRED, invoke

from t2m.bundle import _record
from t2m.dynamic_abi import GridExtent
from t2m.examples import fp8_arithmetic_reference as oracle
from t2m.examples.fp8_reference import FORMATS, decode
from t2m.frontend import inspect_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")
INPUTS = ("x", "y", "lower", "upper", "state")
OUTPUTS = ("add", "sub", "mul", "fma", "none", "all", "div", "sqrt", "raw", "old")


def program(dtype):
    names = (*INPUTS, *OUTPUTS)
    buffers = tuple(TensorBuffer(name, ("n",),
        "readwrite" if name == "state" else "read" if name in INPUTS else "write", dtype) for name in names)
    scalars = (RuntimeScalar("n", "i32", len(names), bounds=(0, 65536)),
        RuntimeScalar("steps", "i32", len(names) + 1, bounds=(0, 3)),
        RuntimeScalar("enabled", "i1", len(names) + 2))
    abi = TensorABI("fp8_math", buffers, grid=(GridExtent("n", 8),), runtime_scalars=scalars)
    args = ",".join([*(f"%{name}:!tt.ptr<{dtype}>" for name in names), "%n:i32,%steps:i32,%enabled:i1"])
    typ = f"tensor<8x{dtype}>"
    ptr = f"tensor<8x!tt.ptr<{dtype}>>"
    lines = [f"module {{tt.func public @fp8_math({args}) {{", '''
      %pid=tt.get_program_id x:i32
      %eight=arith.constant 8:i32
      %base=arith.muli %pid,%eight:i32
      %r=tt.make_range {start=0:i32,end=8:i32}:tensor<8xi32>
      %bases=tt.splat %base:i32->tensor<8xi32>
      %lane=arith.addi %bases,%r:tensor<8xi32>
      %ns=tt.splat %n:i32->tensor<8xi32>
      %live=arith.cmpi slt,%lane,%ns:tensor<8xi32>
      %zero32=arith.constant 0:i32
      %one32=arith.constant 1:i32
    ''', f"%zero=arith.constant dense<0.0>:{typ}"]
    for name in names:
        lines += [f"%p_{name}=tt.splat %{name}:!tt.ptr<{dtype}>->{ptr}",
            f"%q_{name}=tt.addptr %p_{name},%lane:{ptr},tensor<8xi32>"]
    for name in INPUTS[:-1]:
        lines.append(f"%v_{name}=tt.load %q_{name},%live,%zero:{ptr}")
    lines += [f'''%carry=scf.for %i=%zero32 to %steps step %one32 iter_args(%acc=%v_x)->({typ}):i32 {{
      %selected=scf.if %enabled -> ({typ}) {{
        %q=tt.precise_divf %acc,%v_y:{typ}
        %s=tt.precise_sqrt %q:{typ}
        scf.yield %s:{typ}
      }} else {{scf.yield %acc:{typ}}}
      scf.yield %selected:{typ}
    }}''', f"%result_add=arith.addf %carry,%v_y:{typ}", f"%result_sub=arith.subf %carry,%v_y:{typ}",
        f"%result_mul=arith.mulf %carry,%v_y:{typ}", f"%result_fma=math.fma %carry,%v_y,%v_lower:{typ}",
        f"%result_none=tt.clampf %carry,%v_lower,%v_upper,propagateNan=none:{typ}",
        f"%result_all=tt.clampf %carry,%v_lower,%v_upper,propagateNan=all:{typ}",
        f"%result_div=tt.precise_divf %carry,%v_y:{typ}", f"%result_sqrt=tt.precise_sqrt %carry:{typ}",
        f"%result_old=tt.atomic_rmw exch,relaxed,gpu,%q_state,%result_add,%live:({ptr},{typ},tensor<8xi1>)->{typ}"]
    for name in OUTPUTS:
        lines.append(f"tt.store %q_{name},%{'v_x' if name == 'raw' else 'result_' + name},%live:{ptr}")
    return "\n".join([*lines, "tt.return", "}}"]), abi


def arguments(dtype, n, steps, enabled, shift=0):
    x = ((np.arange(n, dtype=np.uint32) // 256 + shift) & 255).astype(np.uint8)
    y = ((np.arange(n, dtype=np.uint32) + 3 * shift) & 255).astype(np.uint8)
    if n < 65536:
        x = ((np.arange(n, dtype=np.uint32) + shift) & 255).astype(np.uint8)
    one = oracle.rounded(oracle.Fraction(1), dtype)
    zeros = 0 if dtype.endswith("FNUZ") else 128
    lower = np.resize(np.asarray([1, 129, 0, zeros, one, one | 128], np.uint8), n)
    upper = np.full(n, oracle.MAX_WORD[dtype], np.uint8)
    state = ((np.arange(n, dtype=np.uint32) * 17 + 7 + shift) & 255).astype(np.uint8)
    arrays = [x, y, lower, upper, state, np.asarray(n, np.int32), np.asarray(steps, np.int32), np.asarray(enabled)]
    expected = [[] for _ in OUTPUTS]
    for a, b, lo, hi in zip(x, y, lower, upper, strict=True):
        current = int(a)
        for _ in range(steps if enabled else 0):
            current = oracle.square_root(oracle.binary(current, int(b), "div", dtype), dtype)
        row = [*(oracle.binary(current, int(b), kind, dtype) for kind in ("add", "sub", "mul")),
            oracle.fused(current, int(b), int(lo), dtype),
            oracle.clamp(current, int(lo), int(hi), dtype, propagate=False),
            oracle.clamp(current, int(lo), int(hi), dtype, propagate=True),
            oracle.binary(current, int(b), "div", dtype), oracle.square_root(current, dtype), int(a)]
        for result, word in zip(expected[:-1], row, strict=True):
            result.append(word)
    outputs = [np.asarray(words, np.uint8) for words in expected[:-1]]
    return arrays, [outputs[0].copy(), *outputs, state.copy()]


def compare(actual, expected, dtype, *, exact=False, loose_zero=False):
    assert (actual.shape, actual.dtype) == (expected.shape, np.uint8)
    if exact:
        np.testing.assert_array_equal(actual, expected)
        return
    for got, want in zip(actual.flat, expected.flat, strict=True):
        if math.isnan(decode(int(want), dtype)):
            assert math.isnan(decode(int(got), dtype))
        elif loose_zero and decode(int(want), dtype) == 0:
            assert decode(int(got), dtype) == 0
        else:
            assert got == want, (dtype, int(got), int(want))


@pytest.mark.parametrize("dtype", tuple(FORMATS))
def test_every_fp8_pair_keeps_rounding_control_reuse_and_exchange(tmp_path, dtype):
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
    initial_arrays, initial_expected = arguments(dtype, 65536, 0, True)
    write_shared_bundle(result, tmp_path / "capture", initial_arrays, initial_expected, work=tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"])
    stage_compile_inputs(tmp_path / "capture", tmp_path / "stage",
        result_mode="execution-owned" if dtype == "f8E4M3FNUZ" else None)
    product = compile(StagedProgram(tmp_path / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=tmp_path / "compiled")
    features = json.loads((tmp_path / "compiled/host/compilation.json").read_text())["features"]
    assert "emulate_fp8_elementwise" in features and "lower_atomic_word_views" in features
    library = _record(tmp_path / "compiled/host/model.so")
    for i, (n, steps, enabled, shift) in enumerate(((65536, 0, True, 0), (65536, 0, False, 7),
            (263, 2, True, 31), (257, 3, True, 9), (1, 1, False, 200), (0, 3, True, 0))):
        arrays, expected = (initial_arrays, initial_expected) if i == 0 else arguments(dtype, n, steps, enabled, shift)
        before = [a.tobytes() for a in arrays]
        actual = invoke(tmp_path, arrays, f"invoke-{i}")
        assert len(actual) == len(expected) == 11
        for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
            compare(got, want, dtype, exact=index in (9, 10), loose_zero=index == 5)
        assert before == [a.tobytes() for a in arrays]
    assert _record(tmp_path / "compiled/host/model.so") == library
    assert not any(p.name in {"inputs.npz", "golden.npy", "extra.npz"}
        for p in (tmp_path / "compiled/frontend").rglob("*"))
    if dtype == "f8E4M3FNUZ":
        from pathlib import Path

        from merlin.compiler.frontend.intake.execution_result_signature import (
            validate_signature,
        )
        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

        from t2m import consumer
        from t2m.process_outcome import observe

        signature = json.loads((tmp_path / "compiled/frontend/signature.json").read_text())
        signature["input_constraints"]["inputs"][0]["shape"][0] = "enabled"
        with pytest.raises(ValueError, match="non-Boolean integer scalars"):
            validate_signature(parse_mlir_text((tmp_path / "compiled/frontend/program.mlir").read_text()), signature)
        for label, message in (("boolean-storage", "canonical source storage"),
                ("dimension-mismatch", "violates original input dimensions")):
            arrays, _ = arguments(dtype, 1, 0, True)
            if label == "boolean-storage":
                arrays[-1].view(np.uint8)[...] = 2
            else:
                arrays[-3] = np.asarray(0, np.int32)
            inputs = tmp_path / (label + ".npz")
            np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
            output = tmp_path / label
            output.mkdir()
            command = [os.environ["T2M_MERLIN_PYTHON"], str(Path(consumer.__file__).with_name("consumer_worker.py")),
                "observe", os.environ["T2M_MERLIN_ROOT"], str(tmp_path / "compiled/frontend"),
                str(tmp_path / "compiled/host"), str(inputs), str(output)]
            outcome = observe(command, output, environment=dict(os.environ, TMPDIR=str(output)), timeout=60)
            assert not outcome["timed_out"] and outcome["returncode"] != 0
            assert message in (output / "stdout.bin").read_text() + (output / "stderr.bin").read_text()
            assert not (output / "invocation-started.json").exists() and not (output / "outputs.npz").exists()
    product.verify()
