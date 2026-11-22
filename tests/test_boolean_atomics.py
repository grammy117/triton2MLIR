"""Boolean source-width effects through the complete public Merlin interface."""

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
    initial = (np.arange(8) + shift) % 2 != 0
    values = (np.arange(n) * 3 + shift // 2) % 2 != 0
    indices = (np.arange(n, dtype=np.int32) * (5 if shift else 3)) % 8
    active = np.arange(n) % 3 != 1
    indices[~active] = -1000
    state = initial.copy()
    history = [np.zeros(n, np.bool_) for _ in OPS]
    for i, operation in enumerate(OPS):
        for lane in range(n):
            if active[lane]:
                position = indices[lane]
                history[i][lane] = state[position]
                # Signless i1 has signed values 0/-1 for signed min/max.
                result = operation_reference(-int(state[position]), -int(values[lane]), operation, 1)
                state[position] = bool(result)
    before, failed = np.zeros(1, np.bool_), np.zeros(1, np.bool_)
    if n:
        before[0] = state[0]
        state[0] = bool(np.count_nonzero(values) % 2)
        failed[0] = state[0]
    return [initial, values, indices, np.asarray(n, np.int32)], [state, *history, before, failed]


def test_boolean_atomics_keep_source_width_old_values_and_guards(tmp_path):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    source, abi, _, _ = program("i8")
    source = source.replace("i8", "i1")
    abi = replace(abi, buffers=tuple(replace(b, dtype="i1") if b.dtype == "i8" else b for b in abi.buffers))
    (tmp_path / "source.ttir").write_text(source)
    result = convert_shared(tmp_path / "source.ttir", abi, tmp_path / "translation",
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile="atomics")
    assert all(row["word_bits"] == 1 for row in result.trace["memory"]["atomic_payload"])
    assert not any(op.name.startswith(("tt.", "tts.", "gpu.", "llvm.")) for op in result.module.walk())
    arrays, expected = arguments(8, 0)
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
    for number, (n, shift) in enumerate(((8, 0), (8, 1), (5, 3), (1, 2), (0, 0))):
        arrays, expected = arguments(n, shift)
        before = [value.tobytes() for value in arrays]
        actual = invoke(tmp_path, arrays, f"invoke-{number}")
        assert len(actual) == len(expected) == 12
        for value, reference in zip(actual, expected, strict=True):
            assert value.shape == reference.shape and value.dtype == reference.dtype
            np.testing.assert_array_equal(value, reference)
        assert [value.tobytes() for value in arrays] == before
    for invalid in (-1, 8):
        arrays, _ = arguments(1, 0)
        arrays[2][0] = invalid
        invoke(tmp_path, arrays, f"bad-{invalid}", error="Triton active memory index outside logical allocation")
    assert _record(tmp_path / "compiled/host/model.so") == library
    product.verify()
