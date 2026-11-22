"""Legal narrow TTIR words retain indexed values through public Merlin calls."""

import os
import signal
from pathlib import Path

import numpy as np
import pytest

from t2m.bundle import _record
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = OPT and NATIVE and all(os.environ.get(key) for key in
    ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing shared/native/Merlin environments required")


def histogram_source(bits):
    return f'''module {{tt.func public @narrow(%x:!tt.ptr<i32>,%out:!tt.ptr<i64>,%n:i32) {{
      %r=tt.make_range {{start=0:i32,end=512:i32}}:tensor<512xi32>
      %nv=tt.splat %n:i32->tensor<512xi32>
      %live=arith.cmpi slt,%r,%nv:tensor<512xi32>
      %three=arith.constant dense<3>:tensor<512xi32>
      %one=arith.constant dense<1>:tensor<512xi32>
      %rem=arith.remsi %r,%three:tensor<512xi32>
      %selected=arith.cmpi ne,%rem,%one:tensor<512xi32>
      %mask=arith.andi %live,%selected:tensor<512xi1>
      %xp=tt.splat %x:!tt.ptr<i32>->tensor<512x!tt.ptr<i32>>
      %xa=tt.addptr %xp,%r:tensor<512x!tt.ptr<i32>>,tensor<512xi32>
      %zero=arith.constant dense<0>:tensor<512xi32>
      %raw=tt.load %xa,%live,%zero:tensor<512x!tt.ptr<i32>>
      %value=arith.trunci %raw:tensor<512xi32> to tensor<512xi{bits}>
      %counts=tt.histogram %value,%mask:tensor<512xi{bits}>->tensor<8xi{bits}>
      %words=arith.extui %counts:tensor<8xi{bits}> to tensor<8xi64>
      %bins=tt.make_range {{start=0:i32,end=8:i32}}:tensor<8xi32>
      %op=tt.splat %out:!tt.ptr<i64>->tensor<8x!tt.ptr<i64>>
      %oa=tt.addptr %op,%bins:tensor<8x!tt.ptr<i64>>,tensor<8xi32>
      tt.store %oa,%words:tensor<8x!tt.ptr<i64>>
      tt.return
    }}}}'''


def gather_source(bits):
    return f'''module {{tt.func public @narrow(%x:!tt.ptr<i32>,%indices:!tt.ptr<i32>,%out:!tt.ptr<i32>) {{
      %r=tt.make_range {{start=0:i32,end=16:i32}}:tensor<16xi32>
      %xp=tt.splat %x:!tt.ptr<i32>->tensor<16x!tt.ptr<i32>>
      %xa=tt.addptr %xp,%r:tensor<16x!tt.ptr<i32>>,tensor<16xi32>
      %values=tt.load %xa:tensor<16x!tt.ptr<i32>>
      %matrix=tt.reshape %values:tensor<16xi32>->tensor<2x8xi32>
      %q=tt.make_range {{start=0:i32,end=8:i32}}:tensor<8xi32>
      %ip=tt.splat %indices:!tt.ptr<i32>->tensor<8x!tt.ptr<i32>>
      %ia=tt.addptr %ip,%q:tensor<8x!tt.ptr<i32>>,tensor<8xi32>
      %raw=tt.load %ia:tensor<8x!tt.ptr<i32>>
      %small=arith.trunci %raw:tensor<8xi32> to tensor<8xi{bits}>
      %lookup=tt.reshape %small:tensor<8xi{bits}>->tensor<2x4xi{bits}>
      %chosen=tt.gather %matrix[%lookup] {{axis=1:i32}}:(tensor<2x8xi32>,tensor<2x4xi{bits}>)->tensor<2x4xi32>
      %flat=tt.reshape %chosen:tensor<2x4xi32>->tensor<8xi32>
      %op=tt.splat %out:!tt.ptr<i32>->tensor<8x!tt.ptr<i32>>
      %oa=tt.addptr %op,%q:tensor<8x!tt.ptr<i32>>,tensor<8xi32>
      tt.store %oa,%flat:tensor<8x!tt.ptr<i32>>
      tt.return
    }}}}'''


def compile_public(root, result, arrays, expected):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    write_shared_bundle(result, root / "capture", arrays, [expected], work=root / "native", python=NATIVE)
    stage_compile_inputs(root / "capture", root / "stage")
    product = compile(StagedProgram(root / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE, clang=os.environ["T2M_MERLIN_CLANG"]), output=root / "compiled")
    product.verify()
    assert not any(path.name in {"inputs.npz", "golden.npy", "extra.npz"}
                   for path in (root / "compiled/frontend").rglob("*"))
    return product


def invoke(root, arrays, index, *, invalid=False):
    from t2m import consumer
    from t2m.process_outcome import observe

    inputs, output = root / f"input-{index}.npz", root / f"invoke-{index}"
    np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
    output.mkdir()
    environment = dict(os.environ, TMPDIR=str(output))
    command = [os.environ["T2M_MERLIN_PYTHON"], str(Path(consumer.__file__).with_name("consumer_worker.py")),
        "observe", os.environ["T2M_MERLIN_ROOT"], str(root / "compiled/frontend"),
        str(root / "compiled/host"), str(inputs), str(output)]
    outcome = observe(command, output, environment=environment, timeout=60)
    if invalid:
        assert outcome["returncode"] == -signal.SIGABRT and not outcome["timed_out"]
        assert b"TTIR tile gather index outside source axis" in (
            (output / "stdout.bin").read_bytes() + (output / "stderr.bin").read_bytes())
        assert (output / "invocation-started.json").is_file()
        assert not (output / "outputs.npz").exists()
        return None
    assert outcome["returncode"] == 0 and not outcome["timed_out"]
    with np.load(output / "outputs.npz", allow_pickle=False) as actual:
        assert actual.files == ["out0"]
        return actual["out0"].copy()


def histogram_arguments(bits, n, shift):
    raw = np.zeros(n, np.int32)
    pattern = np.array([-1, 0, 1, 1, 2, 7, 8, 16, -128, 32767], np.int32)
    raw[-min(n, 30):] = np.resize(np.roll(pattern, shift), min(n, 30))
    expected = [0] * 8
    mask = (1 << bits) - 1
    for lane, word in enumerate(raw):
        value = int(word) & mask
        if bits > 1 and value & (1 << (bits - 1)):
            value -= 1 << bits
        if lane % 3 != 1 and 0 <= value < 8:
            expected[value] = (expected[value] + 1) & mask
    return [raw, np.asarray(n, np.int32)], np.asarray(expected, np.int64)


@pytest.mark.parametrize("profile", ["tensor", "guarded-buffer"])
@pytest.mark.parametrize("bits", [1, 4, 8, 16])
@pytest.mark.parametrize("kind", ["histogram", "gather"])
def test_narrow_indexed_computation_reaches_public_merlin(tmp_path, profile, bits, kind):
    source = tmp_path / "source.ttir"
    source.write_text(histogram_source(bits) if kind == "histogram" else gather_source(bits))
    if kind == "histogram":
        abi = TensorABI("narrow", (TensorBuffer("x", ("n",), dtype="i32"),
            TensorBuffer("out", (8,), "write", "i64")),
            runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 512)),))
        cases = [histogram_arguments(bits, n, shift) for n, shift in ((512, 0), (511, 3), (0, 0), (1, 1))]
    else:
        abi = TensorABI("narrow", (TensorBuffer("x", (16,), dtype="i32"),
            TensorBuffer("indices", (8,), dtype="i32"), TensorBuffer("out", (8,), "write", "i32")))
        cases = []
        for shift in (0, 7, -13):
            x = np.arange(16, dtype=np.int32).reshape(2, 8) * 100003 + shift
            indices = np.zeros((2, 4), np.int32) if bits == 1 else ((np.arange(8).reshape(2, 4) + shift) % 8).astype(np.int32)
            cases.append(([x.reshape(-1), indices.reshape(-1)], np.take_along_axis(x, indices, 1).reshape(-1)))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    assert any(row["opcode"] == "tt." + kind for row in result.trace["source_operations"])
    assert not any(op.name.startswith(("tt.", "tts.", "gpu.", "llvm.")) for op in result.module.walk())
    arrays, expected = cases[0]
    product = compile_public(tmp_path, result, arrays, expected)
    library = _record(tmp_path / "compiled/host/model.so")
    for index, (arrays, expected) in enumerate(cases):
        before = [value.tobytes() for value in arrays]
        actual = invoke(tmp_path, arrays, index)
        np.testing.assert_array_equal(actual, expected)
        assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape)
        assert [value.tobytes() for value in arrays] == before
    if kind == "gather":
        # Preserve upstream's signed index-cast interpretation, including i1.
        invalids = (1,) if bits == 1 else (-1, -8 if bits == 4 else 8)
        for index, invalid in enumerate(invalids):
            arrays = [cases[0][0][0].copy(), cases[0][0][1].copy()]
            arrays[1][0] = invalid
            invoke(tmp_path, arrays, "bad-" + str(index), invalid=True)
    assert _record(tmp_path / "compiled/host/model.so") == library
    product.verify()


@pytest.mark.parametrize("kind", ["histogram", "gather"])
def test_real_triton_narrow_inputs_reach_public_merlin(tmp_path, kind):
    if kind == "histogram":
        from test_histogram import capture

        result = capture(tmp_path, "guarded-buffer", "i8", True, True)
        x = np.resize(np.array([-128, -1, 0, 1, 1, 7, 8, 127], np.int8), 13)
        values = [int(value) for lane, value in enumerate(x) if lane % 3 != 1 and 0 <= int(value) < 8]
        expected = np.bincount(values, minlength=8).astype(np.int32).reshape(1, 8)
        arrays = [x, np.asarray(13, np.int32), np.asarray(1, np.int32)]
    else:
        from test_tile_gather import arguments, capture

        result = capture(tmp_path, "guarded-buffer", 1, index_dtype="i8")
        arrays, expected = arguments(1, 2)
        arrays[1] = arrays[1].astype(np.int8)
    assert any(row["opcode"] == "tt." + kind and any("xi8>" in ty for ty in row["input_types"])
               for row in result.trace["source_operations"])
    compile_public(tmp_path, result, arrays, expected)
    actual = invoke(tmp_path, arrays, 0)
    assert (actual.shape, actual.dtype, actual.tobytes()) == (expected.shape, expected.dtype, expected.tobytes())
