"""Same-base differences retain exact byte words and actual guarded memory."""

import json
import os
import signal

import numpy as np
import pytest
from test_pointer_addresses import REQUIRED

from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing shared/native/Merlin tools required")


def capture(root, width=32, *, bytes_only=False, profile="guarded-buffer"):
    from t2m.examples.pointer_address_kernels import relative_byte_words, relative_words

    kernel = relative_byte_words if bytes_only else relative_words
    if bytes_only:
        buffers = (TensorBuffer("x", (1,), dtype="i32"),
                   *(TensorBuffer(name, ("n",), "write" if name == "out" else "read", "i64")
                     for name in ("left", "right", "out")))
        scalars = (RuntimeScalar("n", "i32", 4, bounds=(0, 31)),)
        constants = {"B": 8}
    else:
        buffers = (TensorBuffer("x", ("capacity",), dtype=f"i{width}"),
                   TensorBuffer("out", ("n",), "write", f"i{width}"),
                   TensorBuffer("distance", ("n",), "write", "i64"))
        scalars = tuple(RuntimeScalar(name, "i32", i, bounds=bounds) for i, (name, bounds) in enumerate((
            ("n", (0, 31)), ("origin", (-256, 256)), ("stride", (-8, 8)),
            ("capacity", (0, 256)), ("steps", (0, 4))), 3))
        constants = {"B": 8, "WIDTH": width}
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, {
        **{buffer.name: "*" + buffer.dtype for buffer in buffers},
        **{scalar.name: scalar.dtype for scalar in scalars},
        **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI(kernel.__name__, buffers, grid=(GridExtent("n", 8),), runtime_scalars=scalars)
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile, address_policy="source-words" if profile == "guarded-buffer" else "structured")


def arguments(width, n=13, origin=128, stride=3, capacity=256, steps=3):
    storage = np.dtype(f"int{width}")
    mask = (1 << width) - 1
    words = [(index * 7919 + (1 << (width - 1))) & mask for index in range(capacity)]
    words = [word - (1 << width) if word >> (width - 1) else word for word in words]
    x = np.asarray(words, storage)
    indices = [origin + (1 if lane % 2 == 0 else -1) * (lane * stride + steps - 1)
               for lane in range(n)]
    if steps:
        assert all(0 <= index < capacity for index in indices)
        values = x[indices].copy()
        distances = [(index - origin) * (width // 8) for index in indices]
    else:
        values, distances = np.full(n, -17, storage), [0] * n
    return [x, *(np.asarray(value, np.int32) for value in (n, origin, stride, capacity, steps))], [
        values, np.asarray(distances, np.int64)]


def byte_arguments(n):
    pattern = [-1, 0, 1, -(1 << 63), (1 << 63) - 1, 1 << 61, -17, 33]
    left = np.resize(np.asarray(pattern, np.int64), n)
    right = np.resize(np.asarray(pattern[::-1], np.int64), n)
    values = [(int(a) - int(b)) & ((1 << 64) - 1) for a, b in zip(left, right, strict=True)]
    expected = np.asarray([word - (1 << 64) if word >> 63 else word for word in values], np.int64)
    return [np.asarray([123], np.int32), left, right, np.asarray(n, np.int32)], [expected]


def narrow_arguments(width, n):
    pattern = [-(1 << 31), (1 << 31) - 1, 1 << 30, -(1 << 30), -1, 0, 1, 9]
    offsets = np.resize(np.asarray(pattern, np.int32), n)
    words = [(int(value) * 3 + 5) & ((1 << 32) - 1) for value in offsets]
    signed = [word - (1 << 32) if word >> 31 else word for word in words]
    expected = np.asarray([word * (width // 8) for word in signed], np.int64)
    return [np.asarray([1], np.dtype(f"int{width}")), offsets, np.asarray(n, np.int32)], [expected]


def qualify_and_replay(root, result, initial, cases):
    from test_mixed_effects import expected, qualify

    from scripts import qualify_spike_effect as target
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    assert any(item["kind"] == "closed-pointer-integer-difference" for item in result.trace["transformations"])
    assert "prov.closed_pointer_addresses" in result.mlir_text
    assert not any(op["opcode"].startswith("gpu.") for op in result.trace["source_operations"])
    args, outputs = initial
    policy = expected("", [{"kind": "words"} for _ in outputs])
    captured, compiled = qualify(result, root, args, policy, outputs)
    library = _record(compiled / "host/model.so")
    for index, (args, outputs) in enumerate(cases):
        caller = root / f"caller-{index}"
        caller.mkdir()
        np.savez(caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(args)})
        np.savez(caller / "reference.npz", **{f"out{i}": value for i, value in enumerate(outputs)})
        (caller / "reference.json").write_text(json.dumps(policy))
        invocation = root / f"invoke-{index}"
        invoke_kernel(compiled, caller / "inputs.npz", caller / "reference.json", invocation,
                      merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(invocation)
        assert _record(invocation / "source/host/model.so") == library
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    receipt = target.qualify(captured, root / "target", **options)
    assert target.verify(root / "target", **options) == receipt
    return compiled, policy


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_actual_pointer_differences_drive_runtime_memory_and_retain_byte_results(tmp_path, width):
    result = capture(tmp_path, width)
    assert "scf.for" in result.mlir_text
    cases = [arguments(width, **options) for options in (
        {"n": 0, "origin": 256, "capacity": 0},
        {"n": 31, "origin": -256, "capacity": 0, "steps": 0},
        {"n": 1, "origin": 7, "stride": -8, "steps": 4},
        {"n": 8, "origin": 64, "stride": -8, "steps": 1},
        {"n": 31, "origin": 128, "stride": -3, "steps": 4})]
    compiled, policy = qualify_and_replay(tmp_path, result, arguments(width), cases)
    from t2m.invocation import invoke_kernel

    # Active memory bounds are still checked after address cancellation. An
    # invalid source access receives no invented numeric reference or success.
    args, outputs = arguments(width, n=1, origin=0, capacity=1, steps=1)
    args[2] = np.asarray(1, np.int32)
    caller = tmp_path / "bad-caller"
    caller.mkdir()
    np.savez(caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(args)})
    np.savez(caller / "reference.npz", **{f"out{i}": value for i, value in enumerate(outputs)})
    (caller / "reference.json").write_text(json.dumps(policy))
    refused = tmp_path / "bounds-refusal"
    with pytest.raises(ValueError, match="actual effect outcome differs"):
        invoke_kernel(compiled, caller / "inputs.npz", caller / "reference.json", refused,
                      merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    outcome = json.loads((refused / "evaluation/outcome.json").read_text())
    assert outcome["returncode"] == -signal.SIGABRT
    assert b"active memory index outside logical allocation" in (refused / "evaluation/stdout.bin").read_bytes()
    assert not (refused / "invocation.json").exists() and not (refused / "evaluation/outputs.npz").exists()


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_cancelled_base_keeps_full_modular_integer_words_in_every_memory_profile(tmp_path, profile):
    result = capture(tmp_path, bytes_only=True, profile=profile)
    qualify_and_replay(tmp_path, result, byte_arguments(13), [byte_arguments(n) for n in (0, 1, 8, 31)])


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_original_i32_increment_wraps_before_pointee_byte_scaling(tmp_path, width):
    from t2m.examples.pointer_address_kernels import relative_narrow_words

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(relative_narrow_words, {
        "x": f"*i{width}", "offsets": "*i32", "out": "*i64", "n": "i32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI(relative_narrow_words.__name__, (TensorBuffer("x", (1,), dtype=f"i{width}"),
        TensorBuffer("offsets", ("n",), dtype="i32"), TensorBuffer("out", ("n",), "write", "i64")),
        grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
                            memory_profile="guarded-buffer", address_policy="source-words")
    assert any(op["opcode"] == "arith.muli" and "tensor<8xi32>" in op["output_types"]
               for op in result.trace["source_operations"])
    assert "arith.extsi" in result.mlir_text
    qualify_and_replay(tmp_path, result, narrow_arguments(width, 13), [narrow_arguments(width, n) for n in (0, 1, 8, 31)])


def test_integer_address_loop_retains_its_source_control_and_closed_pointer_contract(tmp_path):
    from t2m.examples.pointer_address_kernels import carried_integer_address

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(carried_integer_address, {
        "x": "*i32", "out": "*i32", "n": "i32", "capacity": "i32", "steps": "i32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI(carried_integer_address.__name__, (
        TensorBuffer("x", ("capacity",), dtype="i32"), TensorBuffer("out", ("n",), "write", "i32")),
        grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 31)),
            RuntimeScalar("capacity", "i32", 3, bounds=(0, 256)), RuntimeScalar("steps", "i32", 4, bounds=(0, 4))))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
                            memory_profile="guarded-buffer", address_policy="source-words")
    assert any(item["kind"] == "integer-address-region-state" for item in result.trace["transformations"])
    assert "scf.for" in result.mlir_text
    assert not any(item.name in ("tt.ptr_to_int", "tt.int_to_ptr") for item in result.module.walk())
