"""Both private atomic and volatile effects remain executable and checked."""

import json
import os
from copy import deepcopy

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.memory_contract import validate_memory_evidence
from t2m.shared_tensor import parse_shared
from t2m.word_events import print_ir

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
REQUIRED = pytest.mark.skipif(not TOOLS, reason="existing source/native/Merlin tools required")


def implementation(root, dtype, *, allocation=True, sync=True, cross=False, effect="atomic-volatile", memory="guarded-buffer"):
    from t2m.allocation_bindings import PointerArgument, ScalarArgument, convert_allocations
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.atomic_volatile_kernels import atomic_volatile_updates
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = root / "source.ttir"
    source_type = "fp" + dtype[1:] if dtype.startswith("f") else dtype
    source.write_text(capture_ttir(atomic_volatile_updates,
        {**{name: "*" + source_type for name in ("x", "state", "old", "seen")},
         "n": "i32", "steps": "i32", "B": "constexpr", "SYNC": "constexpr", "CROSS": "constexpr"},
        {"B": 8, "SYNC": sync, "CROSS": cross}))
    abi = TensorABI("allocation_atomic_volatile_updates" if allocation else "atomic_volatile_updates",
        (TensorBuffer("x", ("n",), dtype=dtype), TensorBuffer("state", (32,), "readwrite", dtype),
         TensorBuffer("old", ("n",), "write", dtype), TensorBuffer("seen", ("n",), "write", dtype)),
        grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
                                                   RuntimeScalar("steps", "i32", 5, bounds=(0, 3))))
    options = dict(optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile=memory,
        address_policy="source-words", effect_profile=effect, f32_atomic_add="ieee" if dtype == "f32" and effect != "volatile" else None)
    if allocation:
        return convert_allocations(source, abi,
            (*[PointerArgument(name) for name in ("x", "state", "old", "seen")], ScalarArgument("n"), ScalarArgument("steps")),
            root / "translation", source_entry="atomic_volatile_updates", **options)
    return convert_shared(source, abi, root / "translation", **options)


def arguments(dtype, n, steps, shift=0, *, sync=True):
    dt = getattr(np, {"i32": "int32", "i64": "int64", "f32": "float32", "f64": "float64"}[dtype])
    x = np.arange(n, dtype=dt) - 4 + shift
    initial = np.arange(32, dtype=dt) - shift
    if dtype == "i64":
        initial += 1 << 55
    state, old, seen = initial.copy(), np.full(n, -7, dtype=dt), np.full(n, -7, dtype=dt)
    for _ in range(steps):
        for origin in range(0, n, 8):
            for lane in range(origin, min(origin + 8, n)):
                slot = origin // 2 + (lane - origin) % 4 if sync else lane
                old[lane] = state[slot]
                state[slot] += x[lane]
            for lane in range(origin, min(origin + 8, n)):
                seen[lane] = state[origin // 2 + (lane - origin) % 4 if sync else lane]
    return [x, initial, np.asarray(n, np.int32), np.asarray(steps, np.int32)], [state, old, seen]


@REQUIRED
@pytest.mark.parametrize("dtype", ["i32", "i64", "f32", "f64"])
@pytest.mark.parametrize("sync", [False, True])
def test_mixed_effects_keep_native_old_values_and_unused_volatile_observations(tmp_path, dtype, sync):
    from t2m.native import execute_native

    result = implementation(tmp_path, dtype, allocation=False, sync=sync)
    validate_memory_evidence(result.trace, result.auxiliary_artifacts)
    admission = json.loads(result.auxiliary_artifacts["memory-admission.json"])
    assert admission["schema"] == "t2m.memory-admission.v6" and admission["owner_fragment_bits"] == 30
    assert admission["effect_profile"] == "private-invocation-typed-atomics-v2"
    assert admission["volatile_profile"] == "private-invocation-volatile-loads-v1"
    assert len(admission["volatile_adaptations"]) == 3 and len(admission["atomic_adaptations"]) == 1
    phases = [row for row in result.trace["transformations"] if row["kind"] == "synchronous-program-memory-phase"]
    assert len(phases) == (2 if sync else 0)
    assert all(row["contract"]["schema"] == "t2m.synchronous-program-barrier.v4" for row in phases)
    for i, (n, steps) in enumerate(((0, 3), (1, 0), (13, 2), (31, 3))):
        arrays, expected = arguments(dtype, n, steps, i, sync=sync)
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native-{i}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
        assert (tmp_path / f"native-{i}/lowered-llvm.mlir").read_text().count("llvm.load volatile ") == 3


@REQUIRED
@pytest.mark.parametrize("effect", ["ordinary", "volatile", "typed-atomics"])
def test_selecting_only_one_effect_cannot_erase_the_other(tmp_path, effect):
    with pytest.raises(ConversionError) as error:
        implementation(tmp_path, "i64", allocation=False, effect=effect)
    assert error.value.opcode in ("tt.load", "tt.atomic_rmw") and error.value.source_id is not None


@REQUIRED
@pytest.mark.parametrize("component", ["atomic", "volatile"])
def test_a_combined_selection_does_not_require_inventing_an_absent_source_effect(tmp_path, component):
    from test_volatile_memory import consumer_check
    from t2m.native import execute_native
    from t2m.shared_tensor import convert_shared

    if component == "atomic":
        from test_atomic_barriers import implementation as original_implementation, arguments as original_arguments
    else:
        from test_volatile_memory import implementation as original_implementation, arguments as original_arguments
    original_root = tmp_path / "original"
    original_root.mkdir()
    original = (original_implementation(original_root, "i64", allocation=False) if component == "atomic" else
                original_implementation(original_root, "i64"))
    result = convert_shared(original_root / "source.ttir", original.spec, tmp_path / "combined",
        optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="atomic-volatile")
    validate_memory_evidence(result.trace, result.auxiliary_artifacts)
    assert consumer_check(tmp_path, result)["accepted"]
    arrays, expected = (original_arguments("i64", 13, 2, shift=2) if component == "atomic" else
                        original_arguments("i64", 13, 2, 2))
    actual, _ = execute_native(result.mlir_text, arrays, tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"],
                              shape_abi=result.spec.runtime_shapes())
    for a, b in zip(actual, expected, strict=True):
        assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


@pytest.fixture(scope="module", params=["i64", "f32"])
def selected(tmp_path_factory, request):
    if not TOOLS:
        pytest.skip("existing source/native/Merlin tools required")
    root = tmp_path_factory.mktemp("atomic-volatile-" + request.param)
    return root, request.param, implementation(root, request.param, memory="auto")


def test_mixed_effect_allocation_reaches_a_reusable_merlin_library(selected):
    from test_integer_products import replay_library
    from t2m.package_compatibility import kernel_representation_differences

    root, dtype, result = selected
    assert any("admission v6" in difference for difference in kernel_representation_differences(result.trace))
    replay_library(root, result, [arguments(dtype, n, steps, i)
        for i, (n, steps) in enumerate(((0, 3), (1, 0), (13, 2), (31, 3)))])


def test_each_effect_guard_and_pipeline_component_is_checked_by_both_readers(selected):
    from test_volatile_memory import consumer_check
    from t2m.source_capsule import record

    root, dtype, result = selected
    outcomes = []
    for damage in ("schema", "atomic-component", "volatile-component", "numeric", "owners", "word-version", "cleanup", "lost-read"):
        trace, artifacts = deepcopy(result.trace), dict(result.auxiliary_artifacts)
        admission, stage = (json.loads(artifacts[name]) for name in ("memory-admission.json", "word-stage.json"))
        if damage == "schema":
            admission["schema"] = "t2m.memory-admission.v4"
        elif damage == "atomic-component":
            admission["effect_profile"] = trace["memory"]["effect_profile"] = "private-invocation-volatile-loads-v1"
        elif damage == "volatile-component":
            admission["volatile_profile"] = trace["memory"]["volatile_profile"] = "ordinary"
        elif damage == "numeric":
            admission["f32_atomic_add"] = trace["memory"]["f32_atomic_add"] = "ftz" if dtype == "f32" else "invalid"
        elif damage == "owners":
            admission["ownership_protocol"] = "sticky-complemented-program-fragments"
        elif damage == "word-version":
            stage["schema"] = "t2m.source-word-stages.v3"
        elif damage == "cleanup":
            stage["compute"]["native_stage"]["command"].insert(1, "--canonicalize")
        else:
            admission["volatile_adaptations"].pop(1)
            trace["memory"]["volatile_adaptations"] = admission["volatile_adaptations"]
        artifacts["memory-admission.json"], artifacts["word-stage.json"] = json.dumps(admission), json.dumps(stage)
        trace["memory"]["admission_sha256"] = record(artifacts["memory-admission.json"])["sha256"]
        trace["memory"]["word_stage_sha256"] = record(artifacts["word-stage.json"])["sha256"]
        with pytest.raises(ValueError) as producer:
            validate_memory_evidence(trace, artifacts)
        consumer = consumer_check(root, result, artifacts=artifacts, trace=trace)
        assert not consumer["accepted"]
        outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": consumer["error"]})
    (root / "memory-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_mixed_source_phases_cannot_drop_either_primitive_or_selection(selected):
    from test_atomic_barriers import merlin_refusal
    from t2m.source_capsule import record
    from t2m.source_protection import validate
    from t2m.shared_tensor import opcode
    from xdsl.dialects import builtin

    root, _, result = selected
    artifacts = result.auxiliary_artifacts
    outcomes = []
    for damage in ("atomic-selection", "volatile-selection", "downgrade", "atomic-kind", "volatile-flag", "lost-atomic"):
        members = dict(artifacts)
        report = json.loads(members["allocation-protection.json"])
        phase = report["barrier_conversions"][0]["contract"]
        if damage == "atomic-selection":
            phase.pop("atomic_effects")
        elif damage == "volatile-selection":
            phase.pop("volatile_effects")
        elif damage == "downgrade":
            phase["schema"] = "t2m.synchronous-program-barrier.v3"
        else:
            module = parse_shared(members["allocation-protected.ttir"])
            if damage == "atomic-kind":
                op = next(op for op in module.walk() if opcode(op) == "tt.atomic_rmw")
                op.properties["atomic_rmw_op"] = builtin.IntegerAttr(10, builtin.i32)
            elif damage == "volatile-flag":
                op = next(op for op in module.walk() if opcode(op) == "tt.load")
                op.properties["isVolatile"] = builtin.IntegerAttr(False, builtin.i1)
            text = print_ir(module)
            if damage == "lost-atomic":
                text = text.replace('"tt.atomic_rmw"', '"test.identity"')
            members["allocation-protected.ttir"] = text
            report["artifacts"]["protected"] = record(text)
        members["allocation-protection.json"] = json.dumps(report)
        with pytest.raises(ValueError) as producer:
            validate(members, artifacts["allocation-original.ttir"], "atomic_volatile_updates")
        consumer = merlin_refusal(root, members, source_entry="atomic_volatile_updates")
        assert "atomic" in consumer or "volatile" in consumer or "barrier" in consumer
        outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": consumer})
    (root / "source-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


@REQUIRED
def test_combined_effects_do_not_grant_cross_program_ordinary_publication(tmp_path):
    import signal
    from t2m.native import observe_native

    result = implementation(tmp_path, "i64", allocation=False, cross=True)
    arrays, _ = arguments("i64", 13, 2)
    outcome = observe_native(result.mlir_text, arrays, tmp_path / "conflict", python=os.environ["T2M_NATIVE_PYTHON"],
                             shape_abi=result.spec.runtime_shapes())
    assert outcome["returncode"] == -signal.SIGABRT and not outcome["timed_out"]
    assert b"Triton memory conflicts across logical programs" in (
        (tmp_path / "conflict/stdout.bin").read_bytes() + (tmp_path / "conflict/stderr.bin").read_bytes())
    assert not (tmp_path / "conflict/outputs.npz").exists()


def test_actual_serial_target_keeps_both_mixed_effects(selected):
    import sys
    from pathlib import Path
    from test_spike_model import TOOLS as TARGET_TOOLS, Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, _, _ = selected
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        report = Q.qualify(root / "capture", root / "target", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 3
    finally:
        sys.path.remove(scripts)


@REQUIRED
def test_compare_exchange_retains_actual_volatile_reads_and_unmasked_padded_updates(tmp_path):
    import sys
    from pathlib import Path
    from test_integer_products import replay_library
    from test_spike_model import TOOLS as TARGET_TOOLS, Q
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.atomic_volatile_kernels import atomic_volatile_cas
    from t2m.frontend import capture_ttir
    from t2m.native import execute_native
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(atomic_volatile_cas,
        {**{name: "*i64" for name in ("x", "state", "old", "seen")}, "n": "i32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("atomic_volatile_cas", (TensorBuffer("x", ("n",), dtype="i64"),
        TensorBuffer("state", (32,), "readwrite", "i64"), TensorBuffer("old", ("n",), "write", "i64"),
        TensorBuffer("seen", ("n",), "write", "i64")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile="atomic-volatile")
    cases = []
    for i, n in enumerate((0, 1, 13, 31)):
        values = np.arange(n, dtype=np.int64) - n
        initial = np.arange(32, dtype=np.int64) - n
        state = initial.copy()
        for lane in range((n + 7) // 8 * 8):
            value = values[lane] if lane < n else 0
            if state[lane] == value:
                state[lane] = value + 1
        expected = [state, initial[:n].copy(), state[:n].copy()]
        arrays = [values, initial, np.asarray(n, np.int32)]
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native-{i}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
        cases.append((arrays, expected))
    assert cases[-1][1][0][31] == 1  # The inactive source CAS remains a real state update.
    replay_library(tmp_path, result, cases)
    if TARGET_TOOLS:
        scripts = str(Path(__file__).resolve().parents[1] / "scripts")
        sys.path.insert(0, scripts)
        try:
            report = Q.qualify(tmp_path / "capture", tmp_path / "target", comparison="words",
                merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
            assert report["evaluation"]["result_count"] == 3
        finally:
            sys.path.remove(scripts)
