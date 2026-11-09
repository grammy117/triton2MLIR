"""Actual atomic payloads, old values, conflicts and package/consumer custody."""

from copy import deepcopy
import json
import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
OPS = ("add", "and", "or", "xor", "max", "min", "umax", "umin", "exch")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


@pytest.fixture(scope="module")
def optimizer():
    from t2m.upstream import inspect_optimizer
    return inspect_optimizer(OPT)


def capture(root, optimizer, operation="add", width=32, scalar=False, *, masked=True, fresh=False,
            initialize=False, order="relaxed", scope="gpu", effect="readwrite", effect_profile="atomics"):
    from t2m.examples.atomic_kernels import indexed_atomic
    from t2m.frontend import capture_ttir
    dtype = f"i{width}"
    source = root / "source.ttir"
    constants = {"B": 8, "OP": operation, "SCALAR": scalar, "MASK": masked, "FRESH": initialize, "SEM": order, "SCOPE": scope}
    pointer = f"*{'u' if operation in ('umax', 'umin') else 'i'}{width}"
    source.write_text(capture_ttir(indexed_atomic, {"x": pointer, "index": "*i32", "state": pointer, "old": pointer,
        "n": "i32", "m": "i32", **{n: "constexpr" for n in constants}}, constants))
    abi = TensorABI("indexed_atomic", (TensorBuffer("x", ("n",), dtype=dtype), TensorBuffer("index", ("n",), dtype="i32"),
        TensorBuffer("state", ("n" if fresh else "m",), "write" if fresh else effect, dtype=dtype),
        TensorBuffer("old", ("n",), "write", dtype=dtype)), grid=(GridExtent("n", 1 if scalar else 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)), RuntimeScalar("m", "i32", 5, bounds=(0, 37))))
    return convert_shared(source, abi, root / "translation", optimizer=optimizer, memory_profile="guarded-buffer",
                          address_policy="source-words", effect_profile=effect_profile)


def signed(value, width):
    value &= (1 << width) - 1
    return value - (1 << width) if value >= 1 << (width - 1) else value


def operation_reference(a, b, operation, width):
    mask = (1 << width) - 1
    if operation == "add": result = a + b
    elif operation == "and": result = a & b
    elif operation == "or": result = a | b
    elif operation == "xor": result = a ^ b
    elif operation == "max": result = max(a, b)
    elif operation == "min": result = min(a, b)
    elif operation == "umax": result = max(a & mask, b & mask)
    elif operation == "umin": result = min(a & mask, b & mask)
    else: result = b
    return signed(result, width)


def arguments(n, operation, width, *, fresh=False, masked=True, collide=False):
    dtype = getattr(np, f"int{width}")
    x = np.resize(np.array([-1, 5, -(1 << (width - 1)), 0, (1 << (width - 1)) - 1, -19], dtype), n)
    index = np.arange(n, dtype=np.int32) % 3 if collide else np.arange(n, dtype=np.int32)
    enabled = np.arange(n) % 3 != 1 if masked else np.ones(n, np.bool_)
    index[~enabled] = -1000  # Inactive invalid addresses must have no effect.
    incoming = np.resize(np.asarray([-13, 7, (1 << (width - 1)) - 1, -(1 << (width - 1))], dtype), n if fresh else 37)
    state, old = np.zeros(n, dtype) if fresh else incoming.copy(), np.zeros(n, dtype)
    for i in range(n):
        if enabled[i]:
            position = index[i]
            old[i] = state[position]
            state[position] = operation_reference(int(state[position]), int(x[i]), operation, width)
    arrays = [x, index, *([] if fresh else [incoming]), np.asarray(n, np.int32), np.asarray(n if fresh else 37, np.int32)]
    return arrays, [state, old]


def domains(operation, width, **options):
    return [arguments(n, operation, width, **options) for n in (0, 1, 13, 31)]


@pytest.mark.parametrize("operation", OPS)
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_integer_atomic_payload_keeps_masks_and_returned_old_values(tmp_path, optimizer, operation, width, scalar):
    from t2m.native import execute_native
    result = capture(tmp_path, optimizer, operation, width, scalar)
    evidence = json.loads(result.auxiliary_artifacts["memory-admission.json"])
    assert evidence["schema"] == "t2m.memory-admission.v3" and evidence["owner_fragment_bits"] == 30
    assert result.trace["memory"]["atomic_payload"][0]["operation"] == operation
    assert "scf.execute_region" in result.mlir_text and "no_inline" in result.mlir_text
    for i, (arrays, expected) in enumerate(domains(operation, width)):
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE, shape_abi=result.spec.runtime_shapes())
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(value, reference)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("operation", OPS)
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_merlin_executes_atomic_event_once_and_reuses_the_library(tmp_path, optimizer, operation, width, scalar):
    from test_integer_products import replay_library
    result = capture(tmp_path, optimizer, operation, width, scalar)
    replay_library(tmp_path, result, domains(operation, width))


@pytest.mark.parametrize("width", [32, 64])
def test_colliding_atomic_add_keeps_all_updates_and_old_value_chain(tmp_path, optimizer, width):
    from t2m.native import execute_native
    result = capture(tmp_path, optimizer, width=width, masked=False)
    arrays, expected = arguments(31, "add", width, masked=False, collide=True)
    actual, _ = execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=result.spec.runtime_shapes())
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("initialize", [False, True])
def test_atomic_needs_real_completed_initialization(tmp_path, optimizer, initialize):
    from t2m.native import execute_native
    result = capture(tmp_path, optimizer, fresh=True, initialize=initialize)
    arrays, expected = arguments(13, "add", 32, fresh=True)
    if initialize:
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=result.spec.runtime_shapes())
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(value, reference)
    else:
        with pytest.raises(ValueError, match="exit -6"):
            execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=result.spec.runtime_shapes())


@pytest.mark.parametrize("reverse", [False, True])
def test_ordinary_observation_and_foreign_atomic_conflict_in_both_orders(tmp_path, optimizer, reverse):
    from t2m.examples.atomic_kernels import mixed_atomic_observation
    from t2m.frontend import capture_ttir
    from t2m.native import execute_native
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(mixed_atomic_observation, {"state": "*i32", "out": "*i32", "REVERSE": "constexpr"}, {"REVERSE": reverse}))
    abi = TensorABI("mixed_atomic_observation", (TensorBuffer("state", (1,), "readwrite", dtype="i32"),
        TensorBuffer("out", (2,), "write", dtype="i32")), grid=(2,))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=optimizer, memory_profile="guarded-buffer",
                            address_policy="source-words", effect_profile="atomics")
    with pytest.raises(ValueError, match="exit -6"):
        execute_native(result.mlir_text, [np.asarray([2], np.int32)], tmp_path / "native", python=NATIVE)


def test_default_profile_and_readonly_atomic_remain_located_refusals(tmp_path, optimizer):
    for i, options in enumerate(({"effect_profile": "ordinary"}, {"effect": "read"})):
        root = tmp_path / str(i)
        root.mkdir()
        with pytest.raises(ConversionError) as error:
            capture(root, optimizer, **options)
        assert error.value.source_id and "atomic" in error.value.opcode


@pytest.mark.parametrize("memory,address", [("tensor", "structured"), ("buffer", "structured"), ("guarded-buffer", "structured")])
def test_atomic_policy_requires_its_explicit_memory_and_address_boundary(tmp_path, memory, address):
    with pytest.raises(ValueError, match="atomic effects require"):
        convert_shared(tmp_path / "missing", None, tmp_path / "translation", memory_profile=memory,
                        address_policy=address, effect_profile="atomics")
    assert not (tmp_path / "translation").exists()


def test_rehashed_atomic_contract_cannot_change_operation_order_or_guard_protocol(tmp_path, optimizer):
    from t2m.memory_contract import validate_memory_evidence
    from t2m.word_evidence import record
    result = capture(tmp_path, optimizer)
    for change in ("protocol", "fragments", "order", "operation", "lost-old", "scope", "guard-cost"):
        trace = deepcopy(result.trace)
        artifacts = dict(result.auxiliary_artifacts)
        admission = json.loads(artifacts["memory-admission.json"])
        if change == "protocol": admission["ownership_protocol"] = "sticky-complemented-program-fragments"
        elif change == "fragments": admission["owner_fragment_bits"] = 31
        elif change == "guard-cost": admission["guard_storage_bytes_per_output_element"] = 0
        elif change == "lost-old": admission["atomic_adaptations"][0]["payload"] = "unused"
        else:
            access = next(a for a in admission["accesses"] if a["effect"] == "atomic-readwrite")
            access["atomic"][{"order": "source_order", "scope": "source_scope", "operation": "operation"}[change]] = "unknown"
        artifacts["memory-admission.json"] = json.dumps(admission)
        trace["memory"]["admission_sha256"] = record(artifacts["memory-admission.json"])["sha256"]
        with pytest.raises(ValueError):
            validate_memory_evidence(trace, artifacts)


@pytest.mark.parametrize("order", ["relaxed", "acquire", "release", "acq_rel"])
@pytest.mark.parametrize("scope", ["cta", "gpu", "sys"])
def test_source_order_and_scope_survive_the_standard_atomic_refinement(tmp_path, optimizer, order, scope):
    from t2m.native import execute_native
    result = capture(tmp_path, optimizer, order=order, scope=scope)
    info = result.trace["memory"]["atomic_payload"][0]
    assert info["source_order"] == order and info["source_scope"] == scope and info["standard_order"] == "acq_rel"
    arrays, expected = arguments(13, "add", 32)
    actual, _ = execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=result.spec.runtime_shapes())
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("bad", [-1, 37])
def test_active_atomic_bounds_remain_executable(tmp_path, optimizer, bad):
    from t2m.native import execute_native
    result = capture(tmp_path, optimizer, masked=False)
    arrays, _ = arguments(13, "add", 32, masked=False)
    arrays[1][0] = bad
    with pytest.raises(ValueError, match="exit -6"):
        execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=result.spec.runtime_shapes())


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("width", [32, 64])
def test_merlin_collisions_retain_a_valid_host_old_value_history(tmp_path, optimizer, width):
    from test_integer_products import replay_library
    result = capture(tmp_path, optimizer, width=width)
    replay_library(tmp_path, result, domains("add", width, collide=True))


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
def test_atomic_ticket_results_form_a_permutation_without_a_lane_order_assumption(tmp_path, optimizer):
    from t2m.examples.atomic_kernels import atomic_tickets
    from t2m.frontend import capture_ttir
    from t2m.shared_bundle import write_shared_bundle
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.consumer import invoke_merlin_host
    from t2m.bundle import _record
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(atomic_tickets, {"state": "*i32", "out": "*i32", "n": "i32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("atomic_tickets", (TensorBuffer("state", (1,), "readwrite", dtype="i32"),
        TensorBuffer("out", ("n",), "write", dtype="i32")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 31)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=optimizer, memory_profile="guarded-buffer",
                            address_policy="source-words", effect_profile="atomics")
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, [np.array([0], np.int32), np.asarray(1, np.int32)],
        [np.array([1], np.int32), np.array([0], np.int32)], work=tmp_path / "native", python=NATIVE)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, (n, start) in enumerate(((0, 7), (13, -11), (31, 2**31 - 8))):
        inputs = tmp_path / "inputs.npz"
        np.savez(inputs, in0=np.array([start], np.int32), in1=np.asarray(n, np.int32))
        (state, tickets), evaluation = invoke_merlin_host(compiled, inputs, tmp_path / f"invoke{i}", merlin_root=root, python=python)
        np.testing.assert_array_equal(state, np.array([signed(start + n, 32)], np.int32))
        np.testing.assert_array_equal(np.sort(tickets), np.sort(np.array([signed(start + j, 32) for j in range(n)], np.int32)))
        assert evaluation["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
def test_merlin_chains_original_old_values_into_later_atomic_operations(tmp_path, optimizer):
    from t2m.examples.atomic_kernels import atomic_chain
    from t2m.frontend import capture_ttir
    from test_integer_products import replay_library
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(atomic_chain, {"state": "*i32", "out_a": "*i32", "out_b": "*i32", "n": "i32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("atomic_chain", (TensorBuffer("state", ("n",), "readwrite", dtype="i32"),
        TensorBuffer("out_a", ("n",), "write", dtype="i32"), TensorBuffer("out_b", ("n",), "write", dtype="i32")),
        grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=optimizer, memory_profile="guarded-buffer",
                            address_policy="source-words", effect_profile="atomics")
    cases = []
    for n in (0, 1, 13, 31):
        initial = np.arange(n, dtype=np.int32) - 13
        next_ = initial + np.int32(1)
        cases.append(([initial, np.asarray(n, np.int32)], [next_ ^ initial, initial.copy(), next_]))
    replay_library(tmp_path, result, cases)


def test_visible_atomic_demo_retains_the_functional_package(tmp_path, optimizer):
    from t2m.cli import shared_demo
    from t2m.bundle import verify_bundle
    out = tmp_path / "capture"
    summary = shared_demo(out, kernel="atomic-rmw", optimizer=optimizer, python=NATIVE, memory_profile="guarded-buffer",
                          address_policy="source-words", effect_profile="atomics")
    receipt = verify_bundle(out)
    assert summary["native_numerical_check"] == "passed" and receipt["materialized_abi"]["outputs"] == 2
    assert json.loads((out / "memory-admission.json").read_text())["effect_profile"] == "private-invocation-integer-atomics-v1"
