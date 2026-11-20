"""Private atomic updates retain old values and program-local phase ordering."""

import json
import os
import subprocess
from copy import deepcopy

import numpy as np
import pytest

from t2m.program_barriers import atomic_options, atomic_selection

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
REQUIRED = pytest.mark.skipif(not TOOLS, reason="existing source/native/Merlin tools required")


def merlin_refusal(root, members, *, trace=None, source=None):
    """Exercise the selected consumer checkout in its own supported environment."""
    request = root / "consumer-refusal.json"
    request.write_text(json.dumps({"members": members, "trace": trace, "source": source}))
    script = '''import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from merlin.frontends.kernel_allocation_evidence import _checked_protection, checked_allocation_sources
request = json.loads(Path(sys.argv[2]).read_text())
members = {name: value.encode() for name, value in request["members"].items()}
try:
    if request["trace"] is None:
        _checked_protection(members, "atomic_phases")
    else:
        checked_allocation_sources(members.__getitem__, request["trace"], request["source"].encode())
except ValueError as error:
    print(str(error))
else:
    raise AssertionError("consumer accepted the changed atomic phase evidence")
'''
    done = subprocess.run([os.environ["T2M_MERLIN_PYTHON"], "-c", script,
        os.environ["T2M_MERLIN_ROOT"], str(request)], capture_output=True, text=True, timeout=30,
        env=dict(os.environ, TMPDIR=str(root), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.mark.parametrize("selection", [None, {}, {"profile": "ordinary", "f32_atomic_add": None},
    {"profile": "private-invocation-integer-atomics-v1", "f32_atomic_add": "ieee"},
    {"profile": "private-invocation-typed-atomics-v2", "f32_atomic_add": "fast"}])
def test_atomic_phase_policy_requires_the_existing_explicit_effect_contract(selection):
    with pytest.raises(ValueError, match="atomic"):
        atomic_options(selection)


def test_ordinary_selection_cannot_grant_atomic_phase_semantics():
    with pytest.raises(ValueError, match="atomic"):
        atomic_selection("ordinary", None)


def test_selecting_atomics_without_source_atomics_retains_the_ordinary_barrier_contract():
    from test_program_barriers import source
    from t2m.program_barriers import expand
    from t2m.shared_tensor import _preflight, _source_ids

    module = source()
    _preflight(module, source_entry="f", atomics=True)
    reports = expand(module, _source_ids, effect_profile="atomics")
    assert reports[0]["contract"]["schema"] == "t2m.synchronous-program-barrier.v1"
    assert "atomic_effects" not in reports[0]["contract"]


def implementation(root, dtype, *, allocation=True, cross=False):
    from t2m.allocation_bindings import PointerArgument, ScalarArgument, convert_allocations
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.atomic_barrier_kernels import atomic_phases
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = root / "source.ttir"
    source_type = "fp32" if dtype == "f32" else dtype
    source.write_text(capture_ttir(atomic_phases,
        {**{name: "*" + source_type for name in ("x", "state", "old", "seen")},
         "n": "i32", "steps": "i32", "B": "constexpr", "CROSS": "constexpr"},
        {"B": 8, "CROSS": cross}))
    abi = TensorABI("allocation_atomic_phases" if allocation else "atomic_phases",
        (TensorBuffer("x", ("n",), dtype=dtype), TensorBuffer("state", (16,), "readwrite", dtype),
         TensorBuffer("old", ("n",), "write", dtype), TensorBuffer("seen", ("n",), "write", dtype)),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
                         RuntimeScalar("steps", "i32", 5, bounds=(0, 3))), grid=(GridExtent("n", 8),))
    options = dict(optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="atomics" if dtype == "i64" else "typed-atomics",
        f32_atomic_add="ieee" if dtype == "f32" else None)
    if allocation:
        return convert_allocations(source, abi,
            (*[PointerArgument(name) for name in ("x", "state", "old", "seen")],
             ScalarArgument("n"), ScalarArgument("steps")), root / "translation",
            source_entry="atomic_phases", **options)
    return convert_shared(source, abi, root / "translation", **options)


def arguments(dtype, n, steps, *, shift=0):
    dt = np.int64 if dtype == "i64" else np.float32
    x = np.arange(n, dtype=dt) - 4 + shift
    initial = np.arange(16, dtype=dt) - shift
    state = initial.copy()
    old, seen = np.full(n, -7, dtype=dt), np.full(n, -7, dtype=dt)
    for _ in range(steps):
        for origin in range(0, n, 8):
            for lane in range(origin, min(origin + 8, n)):
                slot = origin // 2 + (lane - origin) % 4
                old[lane] = state[slot]
                state[slot] += x[lane]
            for lane in range(origin, min(origin + 8, n)):
                seen[lane] = state[origin // 2 + (lane - origin) % 4]
    return [x, initial, np.asarray(n, np.int32), np.asarray(steps, np.int32)], [state, old, seen]


@REQUIRED
@pytest.mark.parametrize("dtype", ["i64", "f32"])
def test_direct_source_retains_atomic_old_values_across_program_phases(tmp_path, dtype):
    from t2m.native import execute_native

    result = implementation(tmp_path, dtype, allocation=False)
    contracts = [r["program_barrier"] for r in result.trace["source_operations"] if r["opcode"] == "ttg.barrier"]
    assert len(contracts) == 2 and all(c["schema"] == "t2m.synchronous-program-barrier.v2" for c in contracts)
    assert "scf.forall" in result.mlir_text and "scf.execute_region" in result.mlir_text
    assert "ttg." not in result.mlir_text and "gpu." not in result.mlir_text
    for i, (n, steps) in enumerate([(0, 3), (1, 0), (13, 2), (31, 3)]):
        arrays, expected = arguments(dtype, n, steps, shift=i)
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native-{i}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


@pytest.fixture(scope="module", params=["i64", "f32"])
def selected(tmp_path_factory, request):
    if not TOOLS:
        pytest.skip("existing source/native/Merlin tools required")
    root = tmp_path_factory.mktemp("atomic-barriers-" + request.param)
    return root, request.param, implementation(root, request.param)


def test_protected_atomic_phases_reach_a_reusable_merlin_library(selected):
    from test_integer_products import replay_library
    from t2m.package_compatibility import kernel_representation_differences

    root, dtype, result = selected
    report = json.loads(result.auxiliary_artifacts["allocation-protection.json"])
    assert len(report["barrier_conversions"]) == 2
    assert any("t2m.synchronous-program-barrier.v2" in difference
               for difference in kernel_representation_differences(result.trace))
    assert all(row["contract"]["atomic_effects"]["profile"] == result.trace["memory"]["effect_profile"]
               for row in report["barrier_conversions"])
    replay_library(root, result, [arguments(dtype, n, steps, shift=i)
        for i, (n, steps) in enumerate([(0, 3), (1, 0), (13, 2), (31, 3)])])


def test_source_atomic_claims_and_events_are_checked_by_both_readers(selected):
    from t2m.source_capsule import record
    from t2m.source_protection import validate

    root, _, result = selected
    artifacts = result.auxiliary_artifacts
    original = artifacts["allocation-original.ttir"]
    outcomes = []
    for change in ("missing-selection", "downgrade", "scope", "primitive", "lost-atomic", "changed-atomic"):
        report = json.loads(artifacts["allocation-protection.json"])
        members = dict(artifacts)
        if change in ("missing-selection", "downgrade", "scope"):
            contract = report["barrier_conversions"][0]["contract"]
            if change == "missing-selection":
                contract.pop("atomic_effects")
            elif change == "downgrade":
                contract["schema"] = "t2m.synchronous-program-barrier.v1"
            else:
                contract["atomic_effects"]["profile"] = "external-grid-atomics"
        elif change == "primitive":
            row = next(r for r in report["source_graph"]["operations"] if r["opcode"] == "tt.atomic_rmw")
            row["atomic"]["source_order"] = "relaxed"
        else:
            text = members["allocation-protected.ttir"]
            if change == "lost-atomic":
                text = text.replace('"tt.atomic_rmw"', '"test.identity"')
            else:
                from io import StringIO
                from xdsl.dialects import builtin
                from xdsl.printer import Printer
                from t2m.shared_tensor import opcode, parse_shared

                module = parse_shared(text)
                op = next(op for op in module.walk() if opcode(op) == "tt.atomic_rmw")
                op.properties["atomic_rmw_op"] = builtin.IntegerAttr(10, builtin.i32)
                stream = StringIO()
                Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(module)
                text = stream.getvalue()
            assert text != members["allocation-protected.ttir"]
            members["allocation-protected.ttir"] = text
            report["artifacts"]["protected"] = record(text)
        members["allocation-protection.json"] = json.dumps(report)
        with pytest.raises(ValueError) as producer:
            validate(members, original, "atomic_phases")
        consumer = merlin_refusal(root, members)
        assert "barrier" in consumer
        outcomes.append({"change": change, "producer": str(producer.value), "consumer": consumer})
    (root / "source-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_source_phase_selection_cannot_disagree_with_bound_memory(selected):
    from t2m.source_capsule import validate

    root, _, result = selected
    artifacts = result.auxiliary_artifacts
    outcomes = []
    for field, value in (("effect_profile", "ordinary"), ("f32_atomic_add", "ftz")):
        trace = deepcopy(result.trace)
        trace["memory"][field] = value
        with pytest.raises(ValueError, match="barrier atomic selection") as producer:
            validate(trace, artifacts, result.source_ttir)
        consumer = merlin_refusal(root, artifacts, trace=trace, source=result.source_ttir)
        assert "barrier atomic selection" in consumer
        outcomes.append({"field": field, "producer": str(producer.value), "consumer": consumer})
    (root / "effect-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


@REQUIRED
def test_program_barrier_does_not_reset_atomic_cross_program_conflict_guards(tmp_path):
    import signal

    from t2m.native import observe_native

    result = implementation(tmp_path, "i64", allocation=False, cross=True)
    arrays, _ = arguments("i64", 13, 2)
    outcome = observe_native(result.mlir_text, arrays, tmp_path / "conflict",
        python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
    assert outcome["returncode"] == -signal.SIGABRT
    assert b"Triton memory conflicts across logical programs" in (
        (tmp_path / "conflict/stdout.bin").read_bytes() + (tmp_path / "conflict/stderr.bin").read_bytes())
    assert not (tmp_path / "conflict/outputs.npz").exists()
