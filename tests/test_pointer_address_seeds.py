"""Before-region pointer arithmetic retains numeric state through real consumers."""

import json
import os
import signal

import numpy as np
import pytest
from test_pointer_address_seed_contract import seed_source, unaligned_source
from test_pointer_addresses import REQUIRED

from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing native/shared/Merlin tools required")


def capture(root, width=32, tiled=False, *, byte_order=None):
    source = root / "source.ttir"
    source.write_text(unaligned_source() if byte_order else seed_source(width, tiled=tiled))
    buffers = (TensorBuffer("p", ("capacity",), dtype=f"i{width}"),
               *((TensorBuffer("out", (4,), "readwrite", f"i{width}"),
                  TensorBuffer("dist", (4,), "write", "i64")) if tiled else
                 (TensorBuffer("out", (1,), "write", "i64"),)))
    start = len(buffers)
    abi = TensorABI("seed_tile" if tiled else "seed_scalar", buffers, runtime_scalars=(
        RuntimeScalar("count", "i64", start, bounds=(-(1 << 31), 4)),
        RuntimeScalar("capacity", "i32", start + 1, bounds=(0, 32))))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
                          memory_profile="guarded-buffer", address_policy="source-words", byte_order=byte_order)


def arguments(width, tiled, count=4, capacity=32):
    dtype = np.dtype(f"int{width}")
    x = np.asarray([(i * 7 + 3) % 97 - 48 for i in range(capacity)], dtype)
    if not tiled:
        for index, word in ((8, -1), (11, -2), (14, 0), (17, -7)):
            if index < capacity:
                x[index] = word
        value = count
        while value > 0:
            index = (value * 3 + 5) & ((1 << 32) - 1)
            if index >= 1 << 31:
                index -= 1 << 32
            assert 0 <= index < capacity
            value = int(x[index])
        word = (value * 3 + 5) & ((1 << 32) - 1)
        word -= (1 << 32) if word >= 1 << 31 else 0
        arrays = [x, np.asarray(count, np.int64), np.asarray(capacity, np.int32)]
        return arrays, [np.asarray([word * (width // 8)], np.int64)]
    initial = np.asarray([-91, 37, 17, -3], dtype)
    result = x[count:count + 4].copy() if count > 0 else initial.copy()
    assert len(result) == 4
    final_count = min(count, 0)
    offsets = [(final_count + lane) & ((1 << 32) - 1) for lane in range(4)]
    delta = np.asarray([(word - (1 << 32) if word >= 1 << 31 else word) * (width // 8)
                        for word in offsets], np.int64)
    return [x, initial, np.asarray(count, np.int64), np.asarray(capacity, np.int32)], [result, delta]


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("tiled", [False, True])
def test_rebased_before_region_memory_and_relative_results_execute_with_source_widths(tmp_path, width, tiled):
    from test_mixed_effects import expected, qualify

    from scripts import qualify_spike_effect as target
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    result = capture(tmp_path, width, tiled)
    assert any(item["kind"] == "integer-address-seed-rebase" for item in result.trace["transformations"])
    arrays, outputs = arguments(width, tiled)
    policy = expected("", [{"kind": "words"} for _ in outputs])
    captured, compiled = qualify(result, tmp_path, arrays, policy, outputs)
    library = _record(compiled / "host/model.so")
    for index, count in enumerate((0, 1, 2, 3, -(1 << 31))):
        arrays, outputs = arguments(width, tiled, count=count, capacity=0 if count <= 0 else 32)
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(caller / "reference.npz", **{f"out{i}": value for i, value in enumerate(outputs)})
        (caller / "reference.json").write_text(json.dumps(policy))
        invocation = tmp_path / f"invoke-{index}"
        invoke_kernel(compiled, caller / "inputs.npz", caller / "reference.json", invocation,
                      merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(invocation)
        assert _record(invocation / "source/host/model.so") == library
    if width == 32:
        arrays, outputs = arguments(width, tiled, count=0, capacity=0)
        arrays[-2] = np.asarray(1, np.int64)
        caller = tmp_path / "invalid-address"
        caller.mkdir()
        np.savez(caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(caller / "reference.npz", **{f"out{i}": value for i, value in enumerate(outputs)})
        (caller / "reference.json").write_text(json.dumps(policy))
        refused = tmp_path / "bounds-refusal"
        with pytest.raises(ValueError, match="actual effect outcome differs"):
            invoke_kernel(compiled, caller / "inputs.npz", caller / "reference.json", refused,
                          merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert json.loads((refused / "evaluation/outcome.json").read_text())["returncode"] == -signal.SIGABRT
        assert b"active memory index outside logical allocation" in (refused / "evaluation/stdout.bin").read_bytes()
        assert not (refused / "invocation.json").exists() and not (refused / "evaluation/outputs.npz").exists()
        assert _record(compiled / "host/model.so") == library
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    report = target.qualify(captured, tmp_path / "target", **options)
    assert target.verify(tmp_path / "target", **options) == report


def unaligned_arguments(byte_order, count=4, capacity=32, word=-1):
    x = np.full(capacity, word, np.int32)
    final_count = count
    if count > 0:
        raw = b"".join(int(value).to_bytes(4, byte_order, signed=True) for value in x)
        assert len(raw[1:5]) == 4
        final_count = int.from_bytes(raw[1:5], byte_order, signed=True)
        assert final_count <= 0  # finite source executions; no general termination assumption
    return [x, np.asarray(count, np.int64), np.asarray(capacity, np.int32)], [np.asarray([final_count], np.int64)]


@pytest.mark.parametrize("byte_order", ["little", "big"])
def test_existing_unaligned_byte_view_keeps_original_pointer_pair_and_real_word_bounds(tmp_path, byte_order):
    from test_mixed_effects import expected, qualify

    from scripts import qualify_spike_effect as target
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    result = capture(tmp_path, byte_order=byte_order)
    assert not any(item["kind"] == "integer-address-seed-rebase" for item in result.trace["transformations"])
    arrays, outputs = unaligned_arguments(byte_order)
    policy = expected("", [{"kind": "words"}])
    captured, compiled = qualify(result, tmp_path, arrays, policy, outputs)
    library = _record(compiled / "host/model.so")
    for index, (count, capacity, word) in enumerate(((0, 0, -1), (-(1 << 31), 0, -1), (1, 32, -2), (3, 2, -3))):
        arrays, outputs = unaligned_arguments(byte_order, count, capacity, word)
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(caller / "reference.npz", **{f"out{i}": value for i, value in enumerate(outputs)})
        (caller / "reference.json").write_text(json.dumps(policy))
        invocation = tmp_path / f"invoke-{index}"
        invoke_kernel(compiled, caller / "inputs.npz", caller / "reference.json", invocation,
                      merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(invocation)
        assert _record(invocation / "source/host/model.so") == library
    arrays, outputs = unaligned_arguments(byte_order)
    arrays[0], arrays[-1] = arrays[0][:1].copy(), np.asarray(1, np.int32)
    caller = tmp_path / "invalid-word"
    caller.mkdir()
    np.savez(caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)})
    np.savez(caller / "reference.npz", **{f"out{i}": value for i, value in enumerate(outputs)})
    (caller / "reference.json").write_text(json.dumps(policy))
    refused = tmp_path / "bounds-refusal"
    with pytest.raises(ValueError, match="actual effect outcome differs"):
        invoke_kernel(compiled, caller / "inputs.npz", caller / "reference.json", refused,
                      merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert json.loads((refused / "evaluation/outcome.json").read_text())["returncode"] == -signal.SIGABRT
    assert b"active memory index outside logical allocation" in (refused / "evaluation/stdout.bin").read_bytes()
    assert not (refused / "invocation.json").exists() and not (refused / "evaluation/outputs.npz").exists()
    assert _record(compiled / "host/model.so") == library
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    report = target.qualify(captured, tmp_path / "target", **options)
    assert target.verify(tmp_path / "target", **options) == report
