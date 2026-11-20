"""Whole models preserve synchronized atomics, snapshots and all result ports."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_atomic_barriers import TOOLS, implementation

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle


@pytest.fixture(scope="module", params=["i64", "f32"])
def selected(tmp_path_factory, request):
    if not TOOLS or not os.environ.get("T2M_MODEL2MLIR_PYTHON"):
        pytest.skip("existing source/model/native/Merlin tools required")
    root = tmp_path_factory.mktemp("atomic-phase-model-" + request.param)
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_atomic_phase_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture),
        "--dtype", request.param], capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    work = root / "kernel"
    work.mkdir()
    kernel = implementation(work, request.param)
    bindings = [KernelBinding("t2m_demo::atomic_phases", kernel, operands={"x": "x", "state": "state"},
        results=("state", "old", "seen"), scalar_arguments={"steps": steps, **({"n": 0} if empty else {})},
        runtime_arguments={"steps": "steps", **({"n": "n"} if empty else {})},
        runtime_operands={} if empty else {"n": "n"}) for steps, empty in ((2, False), (3, False), (1, True))]
    package = write_model_bundle(capture, bindings, root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, request.param, capture, package


def test_complete_model_retains_parameters_snapshots_and_atomic_results(selected):
    _, _, capture, package = selected
    receipt = verify_bundle(package)
    for name, destination in receipt["parent_artifact_bindings"].items():
        assert (capture / name).read_bytes() == (package / destination).read_bytes()
    with np.load(capture / "goldens.npz", allow_pickle=False) as references, np.load(
        package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(references.files) == len(actual.files) == 13
        for name in references.files:
            assert (actual[name].dtype, actual[name].shape, actual[name].tobytes()) == (
                references[name].dtype, references[name].shape, references[name].tobytes())
    calls = json.loads((package / "kernel-bindings.json").read_text())
    assert len(calls["calls"]) == 3 and len(calls["kernels"]) == 1


def reference(x, state, bias):
    def phases(values, initial, steps):
        bins = initial.copy()
        old, seen = np.full_like(values, -7), np.full_like(values, -7)
        for _ in range(steps):
            for origin in range(0, len(values), 8):
                lanes = range(origin, min(origin + 8, len(values)))
                for lane in lanes:
                    slot = origin // 2 + (lane - origin) % 4
                    old[lane] = bins[slot]
                    bins[slot] += values[lane]
                for lane in lanes:
                    seen[lane] = bins[origin // 2 + (lane - origin) % 4]
        return bins, old, seen

    original = x + bias
    first, old1, seen1 = phases(original, state, 2)
    second, old2, seen2 = phases(original, first, 3)
    empty = np.empty(0, dtype=x.dtype)
    return [second, first, old1, seen1, old2, seen2, original, x, second.copy(), empty, empty.copy(), state, bias]


def invocation(root, dtype, n, index):
    dt = np.int64 if dtype == "i64" else np.float32
    x = np.arange(n, dtype=dt) + index - 8
    state = np.arange(16, dtype=dt) - index
    bias = np.asarray(index - 2, dtype=dt)
    inputs = root / f"inputs-{index}.npz"
    # Original parent manifest order is bias, state, x.
    np.savez(inputs, in0=bias, in1=state, in2=x)
    return inputs, reference(x, state, bias)


def test_changed_shapes_parameters_and_values_reuse_the_original_library(selected):
    root, dtype, _, package = selected
    library = package / "consumer/host/model.so"
    before = _record(library)
    for i, n in enumerate([2, 17, 31]):
        inputs, expected = invocation(root, dtype, n, i)
        _, report = invoke_merlin_host(package / "consumer", inputs, root / f"replay-{i}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        with np.load(root / f"replay-{i}/outputs.npz", allow_pickle=False) as actual:
            assert len(actual.files) == 13
            for j, value in enumerate(expected):
                observed = actual[f"out{j}"]
                assert (observed.dtype, observed.shape, observed.tobytes()) == (value.dtype, value.shape, value.tobytes())
    assert _record(library) == before


def test_actual_serial_target_preserves_complete_atomic_phase_model(selected):
    import sys
    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, dtype, _, package = selected
    inputs, expected = invocation(root, dtype, 31, 8)
    golden = root / "target-reference.npz"
    np.savez(golden, **{f"out{i}": value for i, value in enumerate(expected)})
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        report = Q.qualify(package, root / "target", result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
            inputs=inputs, reference=golden)
        assert report["evaluation"]["result_count"] == 13
    finally:
        sys.path.remove(scripts)


def test_rehashed_atomic_phase_claims_refuse_complete_model_intake(selected):
    from copy import deepcopy
    from test_executed_model_packages import rehash
    from t2m.consumer import check_merlin_interface
    from t2m.source_capsule import record

    root, _, _, package = selected
    paths = [package / name for name in ("capture_receipt.json", "kernel-bindings.json",
        "kernels/0/provenance.json", "kernels/0/allocation-protection.json")]
    original = {path: path.read_bytes() for path in paths}
    outcomes = []
    try:
        for damage in ("missing-selection", "primitive", "memory-profile"):
            protection = json.loads(original[paths[3]])
            provenance = json.loads(original[paths[2]])
            if damage == "missing-selection":
                protection["barrier_conversions"][0]["contract"].pop("atomic_effects")
            elif damage == "primitive":
                atomic = next(row for row in protection["source_graph"]["operations"] if row["opcode"] == "tt.atomic_rmw")
                atomic["atomic"]["source_scope"] = "cta"
            else:
                provenance["memory"]["effect_profile"] = "ordinary"
            paths[3].write_text(json.dumps(protection))
            provenance["source_argument_binding"]["source_capsule"]["artifacts"]["allocation-protection.json"] = record(
                paths[3].read_bytes())
            paths[2].write_text(json.dumps(provenance))
            bindings = json.loads(original[paths[1]])
            bindings["kernels"][next(iter(bindings["kernels"]))] = deepcopy(provenance)
            paths[1].write_text(json.dumps(bindings))
            rehash(package)
            with pytest.raises(ValueError) as producer:
                verify_bundle(package)
            with pytest.raises(ValueError) as consumer:
                check_merlin_interface(package, root / ("refused-" + damage), result_mode="execution-owned",
                    merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
            expected = "atomic" if damage == "memory-profile" else "barrier"
            assert expected in str(producer.value) and expected in str(consumer.value)
            outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": str(consumer.value)})
            for path, contents in original.items():
                path.write_bytes(contents)
    finally:
        for path, contents in original.items():
            path.write_bytes(contents)
    verify_bundle(package)
    (root / "model-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_original_model_shape_domain_still_aborts_before_returned_results(selected):
    import signal
    from t2m.consumer import observe_merlin_host

    root, dtype, _, package = selected
    inputs, _ = invocation(root, dtype, 1, 50)
    work = root / "domain-abort"
    outcome = observe_merlin_host(package / "consumer", inputs, work,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert outcome["returncode"] == -signal.SIGABRT and not outcome["timed_out"]
    assert not (work / "outputs.npz").exists()
