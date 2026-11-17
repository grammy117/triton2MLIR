"""Structured address pairs reach the actual reusable Merlin/target interface."""

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


def capture(root, mode, width):
    from t2m.examples.integer_address_region_kernels import integer_address_state

    buffers = (TensorBuffer("x", ("capacity",), dtype=f"i{width}"),
        TensorBuffer("y", ("capacity",), dtype=f"i{width}"),
        TensorBuffer("state", ("n",), "readwrite", f"i{width}"), TensorBuffer("distance", ("n",), "write", "i64"))
    scalars = tuple(RuntimeScalar(name, "i32", i, bounds=bounds) for i, (name, bounds) in enumerate((
        ("n", (0, 31)), ("capacity", (0, 256)), ("steps", (0, 4)),
        ("origin", (-256, 256)), ("stride", (-8, 8)), ("choice", (-2, 2))), 4))
    source = root / "source.ttir"
    source.write_text(capture_ttir(integer_address_state, {
        **{buffer.name: "*" + buffer.dtype for buffer in buffers}, **{scalar.name: scalar.dtype for scalar in scalars},
        "B": "constexpr", "WIDTH": "constexpr", "MODE": "constexpr"}, {"B": 8, "WIDTH": width, "MODE": mode}))
    abi = TensorABI(integer_address_state.__name__, buffers, grid=(GridExtent("n", 8),), runtime_scalars=scalars)
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
                          memory_profile="guarded-buffer", address_policy="source-words")


def arguments(mode, width, n=13, capacity=128, steps=3, origin=64, stride=3, choice=1):
    dtype = np.dtype(f"int{width}")
    mask = (1 << width) - 1

    def words(count, factor, bias):
        values = [(i * factor + bias) & mask for i in range(count)]
        return np.asarray([value - (1 << width) if value >> (width - 1) else value for value in values], dtype)

    x, y, old = words(capacity, 7919, 1 << (width - 1)), words(capacity, -3571, 17), words(n, 37, -91)
    expected = old.copy().view(np.dtype(f"uint{width}"))
    root = y if mode == "branch" and choice <= 0 else x
    position = origin
    for step in range(steps):
        assert not n or 0 <= position and position + n <= capacity
        for lane in range(n):
            expected[lane] = int(expected[lane]) ^ (int(root[position + lane]) & mask)
        position += stride
        if mode == "alternating":
            root = x if (step + choice) % 2 == 0 else y
    distance = np.full(n, steps * stride * (width // 8), np.int64)
    return [x, y, old, *(np.asarray(value, np.int32) for value in (n, capacity, steps, origin, stride, choice))], [
        expected.view(dtype), distance]


@pytest.mark.parametrize("mode", ["for", "while", "branch", "alternating"])
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_address_state_preserves_regions_effects_relative_results_and_library_reuse(tmp_path, mode, width):
    from test_mixed_effects import expected, qualify

    from scripts import qualify_spike_effect as target
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    result = capture(tmp_path, mode, width)
    reports = [item for item in result.trace["transformations"] if item["kind"] == "integer-address-region-state"]
    assert reports and {item["control"] for item in reports} >= {"scf.while" if mode in ("while", "alternating") else "scf.for"}
    if mode in ("branch", "alternating"):
        assert any(item["control"] == "scf.if" for item in reports)
    args, outputs = arguments(mode, width)
    policy = expected("", [{"kind": "words"}, {"kind": "words"}])
    captured, compiled = qualify(result, tmp_path, args, policy, outputs)
    library = _record(compiled / "host/model.so")
    for index, options in enumerate((
        {"n": 0, "capacity": 0, "origin": 256},
        {"n": 31, "capacity": 0, "origin": -256, "steps": 0},
        {"n": 1, "steps": 4, "origin": 31, "stride": -8, "choice": -1},
        {"n": 8, "steps": 1, "origin": 7, "stride": 0, "choice": 0},
        {"n": 31, "steps": 4, "origin": 80, "stride": -3, "choice": -2},
    )):
        args, outputs = arguments(mode, width, **options)
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(args)})
        np.savez(caller / "reference.npz", **{f"out{i}": value for i, value in enumerate(outputs)})
        (caller / "reference.json").write_text(json.dumps(policy))
        invocation = tmp_path / f"invoke-{index}"
        invoke_kernel(compiled, caller / "inputs.npz", caller / "reference.json", invocation,
                      merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(invocation)
        assert _record(invocation / "source/host/model.so") == library
    if width == 32:
        # Reuse the actual library, change an active address outside its root,
        # and require the executable memory guard to abort without a receipt.
        args, outputs = arguments(mode, width, n=1, capacity=1, steps=1, origin=0)
        args[6] = np.asarray(1, np.int32)  # runtime origin == capacity
        caller = tmp_path / "invalid-address"
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
        assert _record(compiled / "host/model.so") == library
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    report = target.qualify(captured, tmp_path / "target", **options)
    assert target.verify(tmp_path / "target", **options) == report


def test_while_independent_ports_execute_the_original_memory_control(tmp_path):
    from test_integer_address_region_contract import while_source
    from test_mixed_effects import expected, qualify

    from scripts import qualify_spike_effect as target
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    source = tmp_path / "source.ttir"
    source.write_text(while_source())
    abi = TensorABI("while_ports", (TensorBuffer("p", (1,), dtype="i32"),
                    TensorBuffer("out", (1,), "write", "i64")),
                    runtime_scalars=(RuntimeScalar("count", "i64", 2, bounds=(0, 4)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
                            memory_profile="guarded-buffer", address_policy="source-words")
    assert any(item["kind"] == "integer-address-region-state" for item in result.trace["transformations"])
    policy = expected("", [{"kind": "words"}])
    captured, compiled = qualify(result, tmp_path, [np.asarray([-1], np.int32), np.asarray(4, np.int64)],
                                  policy, [np.asarray([0], np.int64)])
    library = _record(compiled / "host/model.so")
    # The source uses the loaded signed word as its next condition input.
    # Nonpositive words terminate; no assumption of general termination is made.
    for index, (word, count) in enumerate(((123, 0), (0, 1), (-(1 << 31), 4))):
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(caller / "inputs.npz", in0=np.asarray([word], np.int32), in1=np.asarray(count, np.int64))
        np.savez(caller / "reference.npz", out0=np.asarray([0], np.int64))
        (caller / "reference.json").write_text(json.dumps(policy))
        invocation = tmp_path / f"invoke-{index}"
        invoke_kernel(compiled, caller / "inputs.npz", caller / "reference.json", invocation,
                      merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(invocation)
        assert _record(invocation / "source/host/model.so") == library
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    report = target.qualify(captured, tmp_path / "target", **options)
    assert target.verify(tmp_path / "target", **options) == report
