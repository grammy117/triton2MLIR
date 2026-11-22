"""Canonical unpacked four-bit cells through both public Merlin result ABIs."""

import os
from dataclasses import replace

import numpy as np
import pytest
from test_atomic_memory import operation_reference
from test_narrow_atomics import OPS, REQUIRED, invoke, program

from t2m.bundle import _record
from t2m.shared_tensor import convert_shared

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")


def arguments(n, shift):
    state = (np.asarray([0, 7, 8, 15, 1, 2, 14, 9], np.uint8) + shift) & 15
    values = ((np.resize(np.asarray([1, 15, 7, 8, 0, 9, 3, 13], np.uint8), n)
               + shift) & 15).astype(np.uint8)
    indices = (np.arange(n, dtype=np.int32) * (5 if shift else 3)) % 8
    active = np.arange(n) % 3 != 1
    indices[~active] = -1000
    initial = state.copy()
    history = [np.zeros(n, np.uint8) for _ in OPS]

    def signed(value):
        return int(value) - 16 if value & 8 else int(value)

    for i, operation in enumerate(OPS):
        for lane in range(n):
            if active[lane]:
                position = indices[lane]
                history[i][lane] = state[position]
                state[position] = operation_reference(signed(state[position]), signed(values[lane]), operation, 4) & 15
    before, failed = np.zeros(1, np.uint8), np.zeros(1, np.uint8)
    if n:
        before[0] = state[0]
        state[0] = sum(int(value) for value in values) & 15
        failed[0] = state[0]
    return [initial, values, indices, np.asarray(n, np.int32)], [state, *history, before, failed]


def reject_carrier(root, arrays, name):
    from pathlib import Path

    from t2m import consumer
    from t2m.process_outcome import observe

    inputs, output = root / (name + ".npz"), root / name
    np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
    output.mkdir()
    command = [os.environ["T2M_MERLIN_PYTHON"], str(Path(consumer.__file__).with_name("consumer_worker.py")),
        "observe", os.environ["T2M_MERLIN_ROOT"], str(root / "compiled/frontend"), str(root / "compiled/host"),
        str(inputs), str(output)]
    outcome = observe(command, output, environment=dict(os.environ, TMPDIR=str(output)), timeout=60)
    assert not outcome["timed_out"] and outcome["returncode"] != 0
    assert b"canonical uint8 cells in 0..15" in (output / "stdout.bin").read_bytes() + (output / "stderr.bin").read_bytes()
    assert not (output / "invocation-started.json").exists()
    assert not (output / "outputs.npz").exists()


@pytest.mark.parametrize("result_mode", [None, "execution-owned"])
def test_i4_cells_keep_source_updates_old_values_and_checked_transport(tmp_path, result_mode):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    source, abi, _, _ = program("i8")
    source = source.replace("i8", "i4")
    abi = replace(abi, buffers=tuple(replace(b, dtype="i4") if b.dtype == "i8" else b for b in abi.buffers))
    (tmp_path / "source.ttir").write_text(source)
    result = convert_shared(tmp_path / "source.ttir", abi, tmp_path / "translation",
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile="atomics")
    assert all(row["word_bits"] == 4 for row in result.trace["memory"]["atomic_payload"])
    assert not any(op.name.startswith(("tt.", "tts.", "gpu.", "llvm.")) for op in result.module.walk())
    arrays, expected = arguments(8, 0)
    write_shared_bundle(result, tmp_path / "capture", arrays, expected, work=tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"])
    import json

    compatibility = json.loads((tmp_path / "capture/compatibility.json").read_text())
    assert "unpacked canonical uint8 cell" in json.dumps(compatibility)
    assert "paired packed-nibble mapping" in json.dumps(compatibility)
    stage_compile_inputs(tmp_path / "capture", tmp_path / "stage", result_mode=result_mode)
    product = compile(StagedProgram(tmp_path / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=tmp_path / "compiled")
    product.verify()
    library = _record(tmp_path / "compiled/host/model.so")
    assert not any(p.name in {"inputs.npz", "golden.npy", "extra.npz"}
        for p in (tmp_path / "compiled/frontend").rglob("*"))
    for number, (n, shift) in enumerate([*((8, i) for i in range(16)), (5, 19), (1, 6), (0, 0)]):
        arrays, expected = arguments(n, shift)
        before = [value.tobytes() for value in arrays]
        actual = invoke(tmp_path, arrays, f"invoke-{number}")
        assert len(actual) == len(expected) == 12
        for value, reference in zip(actual, expected, strict=True):
            assert (value.shape, value.dtype, value.tobytes()) == (reference.shape, reference.dtype, reference.tobytes())
            assert np.all(value <= 15)
        assert [value.tobytes() for value in arrays] == before
    for invalid in (-1, 8):
        arrays, _ = arguments(1, 0)
        arrays[2][0] = invalid
        invoke(tmp_path, arrays, f"bad-index-{invalid}", error="Triton active memory index outside logical allocation")
    for i, invalid in enumerate((16, 255)):
        arrays, _ = arguments(8, 0)
        arrays[i][0] = invalid
        reject_carrier(tmp_path, arrays, f"bad-carrier-{i}")
    arrays, expected = arguments(1, 0)
    arrays[0][0] = 16
    with pytest.raises(ValueError, match="canonical uint8 cells in 0..15"):
        write_shared_bundle(result, tmp_path / "bad-capture", arrays, expected,
            work=tmp_path / "bad-native", python=os.environ["T2M_NATIVE_PYTHON"])
    assert not (tmp_path / "bad-capture").exists()
    if result_mode is None:
        from t2m.native import execute_native

        with pytest.raises(ValueError, match="canonical uint8 cells in 0..15"):
            execute_native(result.mlir_text, arrays, tmp_path / "bad-native-direct",
                python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=abi.runtime_shapes())
        assert not (tmp_path / "bad-native-direct/outputs.npz").exists()
    assert _record(tmp_path / "compiled/host/model.so") == library
    product.verify()
