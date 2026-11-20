"""Volatile loads retain primitive semantics and executable program phases."""

import json
import os
from copy import deepcopy

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.program_barriers import volatile_options, volatile_selection

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
REQUIRED = pytest.mark.skipif(not TOOLS, reason="existing source/native/Merlin tools required")


@pytest.mark.parametrize("selection", [None, {}, {"profile": "ordinary"},
    {"profile": "private-invocation-volatile-loads-v1", "external_participants": True}])
def test_volatile_phase_policy_does_not_acquire_external_or_ordinary_semantics(selection):
    with pytest.raises(ValueError, match="volatile"):
        volatile_options(selection)


def test_no_volatile_source_retains_the_existing_ordinary_barrier_contract():
    from test_program_barriers import source
    from t2m.program_barriers import expand
    from t2m.shared_tensor import _preflight, _source_ids

    module = source()
    _preflight(module, source_entry="f", volatile=True)
    reports = expand(module, _source_ids, effect_profile="volatile")
    assert reports[0]["contract"]["schema"] == "t2m.synchronous-program-barrier.v1"
    with pytest.raises(ValueError, match="volatile"):
        volatile_selection("ordinary")


def test_shared_original_location_names_do_not_conflate_distinct_volatile_reads():
    from t2m.barrier_protection import validate
    from t2m.program_barriers import expand
    from t2m.shared_tensor import _preflight, _source_ids, parse_shared
    from t2m.source_graph import inventory
    from t2m.word_events import print_ir

    module = parse_shared('''"builtin.module"() ({
      "tt.func"() <{sym_name="f",function_type=(!tt.ptr<i64>)->()}> ({^bb0(%p:!tt.ptr<i64>):
        %a="tt.load"(%p) <{isVolatile=true}>:(!tt.ptr<i64>)->i64 loc("shared-original-location")
        "ttg.barrier"() <{addrSpace=31:i32}>:()->()
        %b="tt.load"(%p) <{isVolatile=true}>:(!tt.ptr<i64>)->i64 loc("shared-original-location")
        "tt.return"():()->()
      }):()->()
    }):()->()''')
    generic = print_ir(module)
    records = _preflight(module, source_entry="f", volatile=True)
    graph = inventory(module, records)
    reports = expand(module, _source_ids, effect_profile="volatile")
    validate(graph, reports, generic, print_ir(module), source_entry="f")


def implementation(root, dtype, *, allocation=True, cross=False, effect="volatile"):
    from t2m.allocation_bindings import PointerArgument, ScalarArgument, convert_allocations
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.volatile_barrier_kernels import volatile_exchange
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = root / "source.ttir"
    source_type = "fp" + dtype[1:] if dtype.startswith("f") else dtype
    source.write_text(capture_ttir(volatile_exchange,
        {"state": "*" + source_type, "out": "*" + source_type, "n": "i32", "steps": "i32",
         "B": "constexpr", "CROSS": "constexpr"}, {"B": 4, "CROSS": cross}))
    abi = TensorABI("allocation_volatile_exchange" if allocation else "volatile_exchange",
        (TensorBuffer("state", ("n", 4), "readwrite", dtype), TensorBuffer("out", ("n", 4), "write", dtype)),
        grid=(GridExtent("n", 4),), runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 19)),
                                                   RuntimeScalar("steps", "i32", 3, bounds=(0, 4))))
    options = dict(optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile="guarded-buffer",
                   address_policy="source-words", effect_profile=effect)
    if allocation:
        return convert_allocations(source, abi,
            (PointerArgument("state"), PointerArgument("out"), ScalarArgument("n"), ScalarArgument("steps")),
            root / "translation", source_entry="volatile_exchange", **options)
    return convert_shared(source, abi, root / "translation", **options)


def arguments(dtype, n, steps, shift=0):
    dt = getattr(np, {"i32": "int32", "i64": "int64", "f32": "float32", "f64": "float64"}[dtype])
    initial = np.arange(n * 4, dtype=dt).reshape(n, 4) + shift
    if dtype == "i64":
        initial += 1 << 55
    state = initial + steps * (steps + 1) // 2
    observed = np.full((n, 4), -7, dtype=dt)
    if steps:
        for origin in range(0, n, 4):
            size = min(4, n - origin)
            tile = np.full((4, 4), -3, dtype=dt)
            tile[:size] = state[origin:origin + size]
            observed[origin:origin + size] = tile.T[:size]
    return [initial, np.asarray(n, np.int32), np.asarray(steps, np.int32)], [state, observed]


@REQUIRED
@pytest.mark.parametrize("dtype", ["i32", "i64", "f32", "f64"])
def test_real_cross_lane_volatile_phases_preserve_native_words_and_unused_observations(tmp_path, dtype):
    from t2m.native import execute_native

    result = implementation(tmp_path, dtype, allocation=False)
    reports = [row for row in result.trace["transformations"] if row["kind"] == "synchronous-program-memory-phase"]
    assert len(reports) == 2
    assert all(row["contract"]["schema"] == "t2m.synchronous-program-barrier.v3" for row in reports)
    assert len(result.trace["memory"]["volatile_adaptations"]) == 3
    assert "scf.forall" in result.mlir_text and "scf.execute_region" in result.mlir_text
    assert "ttg." not in result.mlir_text and "gpu." not in result.mlir_text
    for i, (n, steps) in enumerate([(0, 3), (1, 0), (7, 2), (19, 4)]):
        arrays, expected = arguments(dtype, n, steps, i)
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native-{i}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
        assert (tmp_path / f"native-{i}/lowered-llvm.mlir").read_text().count("llvm.load volatile ") == 3


@REQUIRED
def test_default_effect_selection_still_refuses_volatile_program_reads(tmp_path):
    with pytest.raises(ConversionError, match="volatile observations"):
        implementation(tmp_path, "i64", allocation=False, effect="ordinary")


@pytest.fixture(scope="module", params=["i64", "f32"])
def selected(tmp_path_factory, request):
    if not TOOLS:
        pytest.skip("existing source/native/Merlin tools required")
    root = tmp_path_factory.mktemp("volatile-phases-" + request.param)
    return root, request.param, implementation(root, request.param)


def test_protected_volatile_phases_reach_a_reusable_merlin_library(selected):
    from test_integer_products import replay_library
    from t2m.package_compatibility import kernel_representation_differences

    root, dtype, result = selected
    protection = json.loads(result.auxiliary_artifacts["allocation-protection.json"])
    assert len(protection["barrier_conversions"]) == 2
    assert any("t2m.synchronous-program-barrier.v3" in text
               for text in kernel_representation_differences(result.trace))
    replay_library(root, result, [arguments(dtype, n, steps, i)
        for i, (n, steps) in enumerate([(0, 3), (1, 0), (7, 2), (19, 4)])])


def test_source_volatile_flags_profiles_and_phases_are_checked_by_both_readers(selected):
    from test_atomic_barriers import merlin_refusal
    from t2m.source_capsule import record
    from t2m.source_protection import validate
    from t2m.shared_tensor import opcode, parse_shared
    from t2m.word_events import print_ir
    from xdsl.dialects import builtin

    root, _, result = selected
    artifacts = result.auxiliary_artifacts
    outcomes = []
    for damage in ("missing-selection", "downgrade", "scope", "primitive", "lost-flag", "lost-read", "phase-order"):
        members = dict(artifacts)
        report = json.loads(members["allocation-protection.json"])
        phase = report["barrier_conversions"][0]["contract"]
        if damage == "missing-selection":
            phase.pop("volatile_effects")
        elif damage == "downgrade":
            phase["schema"] = "t2m.synchronous-program-barrier.v1"
        elif damage == "scope":
            phase["volatile_effects"]["profile"] = "external-volatile-loads"
        elif damage == "primitive":
            row = next(row for row in report["source_graph"]["operations"] if "volatile_read" in row)
            row["volatile_read"]["observation"] = "stable-tensor-read"
        else:
            module = parse_shared(members["allocation-protected.ttir"])
            read = next(op for op in module.walk() if opcode(op) == "tt.load")
            if damage == "lost-flag":
                read.properties["isVolatile"] = builtin.IntegerAttr(False, builtin.i1)
            elif damage == "phase-order":
                phases = [op for op in module.walk() if op.name == "scf.execute_region"]
                block = phases[0].parent_block()
                block.detach_op(phases[1])
                block.insert_op_before(phases[1], phases[0])
            text = print_ir(module)
            if damage == "lost-read":
                text = text.replace('"tt.load"', '"test.identity"', 1)
            assert text != members["allocation-protected.ttir"]
            members["allocation-protected.ttir"] = text
            report["artifacts"]["protected"] = record(text)
        members["allocation-protection.json"] = json.dumps(report)
        with pytest.raises(ValueError) as producer:
            validate(members, artifacts["allocation-original.ttir"], "volatile_exchange")
        consumer = merlin_refusal(root, members, source_entry="volatile_exchange")
        assert "barrier" in consumer or "volatile" in consumer
        outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": consumer})
    (root / "source-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_source_phase_profile_must_match_bound_volatile_memory(selected):
    from test_atomic_barriers import merlin_refusal
    from t2m.source_capsule import validate

    root, _, result = selected
    trace = deepcopy(result.trace)
    trace["memory"]["effect_profile"] = "ordinary"
    with pytest.raises(ValueError, match="barrier volatile selection") as producer:
        validate(trace, result.auxiliary_artifacts, result.source_ttir)
    consumer = merlin_refusal(root, result.auxiliary_artifacts, trace=trace, source=result.source_ttir)
    assert "barrier volatile selection" in consumer
    (root / "effect-refusal.json").write_text(json.dumps({"producer": str(producer.value), "consumer": consumer}) + "\n")


def test_actual_serial_target_retains_the_cross_lane_volatile_phases(selected):
    import sys
    from pathlib import Path
    from test_spike_model import TOOLS as TARGET_TOOLS, Q
    from t2m.shared_bundle import write_shared_bundle

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, dtype, result = selected
    arrays, expected = arguments(dtype, 19, 3)
    bundle = root / "target-capture"
    write_shared_bundle(result, bundle, arrays, expected, work=root / "target-native", python=os.environ["T2M_NATIVE_PYTHON"])
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        report = Q.qualify(bundle, root / "target", comparison="words", merlin_root=os.environ["T2M_MERLIN_ROOT"],
                           python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
    finally:
        sys.path.remove(scripts)


@REQUIRED
def test_volatile_barrier_cannot_authorize_cross_program_publication(tmp_path):
    import signal
    from t2m.native import observe_native

    result = implementation(tmp_path, "i64", allocation=False, cross=True)
    arrays, _ = arguments("i64", 8, 1)
    outcome = observe_native(result.mlir_text, arrays, tmp_path / "conflict", python=os.environ["T2M_NATIVE_PYTHON"],
                             shape_abi=result.spec.runtime_shapes())
    assert outcome["returncode"] == -signal.SIGABRT and not outcome["timed_out"]
    assert b"Triton memory conflicts across logical programs" in (
        (tmp_path / "conflict/stdout.bin").read_bytes() + (tmp_path / "conflict/stderr.bin").read_bytes())
    assert not (tmp_path / "conflict/outputs.npz").exists()
