"""Complete models keep early exits, synchronized updates and old snapshots."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from test_cfg_barriers import TOOLS, capture

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle


@pytest.fixture(scope="module", params=[("i64", "ordinary"), ("f32", "atomic-volatile")])
def selected(tmp_path_factory, request):
    if not TOOLS or not os.environ.get("T2M_MODEL2MLIR_PYTHON"):
        pytest.skip("existing source/model/native/Merlin tools required")
    dtype, effect = request.param
    root = tmp_path_factory.mktemp("cfg-barrier-model-" + dtype)
    parent = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_cfg_barrier_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(parent),
        "--dtype", dtype, "--effect-profile", effect], capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    work = root / "kernel"
    work.mkdir()
    kernel = capture(work, dtype, effect, allocation=True)
    bindings = [KernelBinding("t2m_demo::cfg_program_exchange", kernel,
        operands={name: name for name in ("state", "old", "seen")}, results=("state", "old", "seen"),
        scalar_arguments={"steps": steps, "mode": mode, **({"n": 0} if empty else {})},
        runtime_arguments={"steps": "steps", "mode": "mode", **({"n": "n"} if empty else {})},
        runtime_operands={} if empty else {"n": "n"}) for steps, mode, empty in ((2, 0, False), (3, 1, False), (1, 1, True))]
    package = write_model_bundle(parent, bindings, root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, dtype, parent, package


def words(path, expected):
    with np.load(path, allow_pickle=False) as actual:
        assert len(actual.files) == len(expected) == 12
        for index, b in enumerate(expected):
            a = actual[f"out{index}"]
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


def reference(x, bias):
    def exchange(state, old, seen, steps, mode):
        state, old, seen = state.copy(), old.copy(), seen.copy()
        for origin in range(0, len(state), 4):
            if (origin // 4 + mode) % 2 == 0:
                continue
            stop = min(origin + 4, len(state))
            for step in range(steps):
                old[origin:stop] = state[origin:stop]
                state[origin:stop] += step + 1
                tile = np.full((4, 4), -3, dtype=state.dtype)
                tile[:stop - origin] = state[origin:stop]
                seen[origin:stop] = tile.T[:stop - origin]
        return state, old, seen

    original = x + bias
    first, a, b = exchange(original, np.full_like(x, -17), np.full_like(x, -23), 2, 0)
    second, c, d = exchange(first, a, b, 3, 1)
    empty, e, f = exchange(x[:0], a[:0], b[:0], 1, 1)
    return [second, first, a, b, c, d, original, x, empty, e, f, bias]


def invocation(root, dtype, rows, index):
    dt = np.int64 if dtype == "i64" else np.float32
    x = np.arange(rows * 4, dtype=dt).reshape(rows, 4) - index - 9
    if dtype == "i64":
        x += 1 << 55
    bias = np.asarray([index, -2, 3, -4], dtype=dt)
    inputs = root / f"inputs-{index}.npz"
    np.savez(inputs, in0=bias, in1=x)
    return inputs, reference(x, bias)


def test_complete_model_retains_all_twelve_results_and_original_archives(selected):
    _, _, parent, package = selected
    receipt = verify_bundle(package)
    for name, target in receipt["parent_artifact_bindings"].items():
        assert (parent / name).read_bytes() == (package / target).read_bytes()
    with np.load(parent / "goldens.npz", allow_pickle=False) as original:
        words(package / "consumer/host/outputs.npz", [original[f"out{i}"].copy() for i in range(12)])
    protection = json.loads((package / "kernels/0/allocation-protection.json").read_text())
    assert all(phase["contract"]["source_control"] == "whole-program-scalar-cfg-v1"
               for phase in protection["barrier_conversions"])


def test_changed_model_shapes_and_parameters_reuse_the_library(selected):
    root, dtype, _, package = selected
    library = _record(package / "consumer/host/model.so")
    for index, rows in enumerate((2, 11, 19)):
        inputs, expected = invocation(root, dtype, rows, index)
        actual, observation = invoke_merlin_host(package / "consumer", inputs, root / f"replay-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert observation["functional_inputs_preserved"]
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def test_source_model_domain_guard_still_aborts_before_publishing_results(selected):
    from t2m.consumer import observe_merlin_host

    root, dtype, _, package = selected
    inputs, _ = invocation(root, dtype, 1, 50)
    work = root / "domain-abort"
    outcome = observe_merlin_host(package / "consumer", inputs, work,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert outcome["returncode"] == -6 and not outcome["timed_out"]
    assert not (work / "outputs.npz").exists()


def test_rehashed_model_phase_claims_cannot_change_original_cfg_participation(selected):
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
        for damage in ("missing-control", "different-control", "grid-scope"):
            protection = json.loads(original[paths[3]])
            for phase in protection["barrier_conversions"]:
                if damage == "missing-control":
                    phase["contract"].pop("source_control")
                elif damage == "different-control":
                    phase["contract"]["source_control"] = "per-lane"
                else:
                    phase["contract"]["participants"] = "all-programs"
            contracts = {phase["contract"]["source_id"]: phase["contract"] for phase in protection["barrier_conversions"]}
            for row in protection["source_graph"]["operations"]:
                if row["id"] in contracts:
                    row["program_barrier"] = deepcopy(contracts[row["id"]])
            paths[3].write_text(json.dumps(protection))
            provenance = json.loads(original[paths[2]])
            provenance["source_argument_binding"]["source_capsule"]["artifacts"]["allocation-protection.json"] = record(paths[3].read_bytes())
            for row in provenance["transformations"]:
                if row["kind"] == "protected-allocation-program-memory-phases":
                    row["phases"] = deepcopy(protection["barrier_conversions"])
            paths[2].write_text(json.dumps(provenance))
            bindings = json.loads(original[paths[1]])
            bindings["kernels"][next(iter(bindings["kernels"]))] = deepcopy(provenance)
            paths[1].write_text(json.dumps(bindings))
            rehash(package)
            with pytest.raises(ValueError, match="barrier") as producer:
                verify_bundle(package)
            with pytest.raises(ValueError, match="barrier") as consumer:
                check_merlin_interface(package, root / ("refused-" + damage), result_mode="execution-owned",
                    merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
            outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": str(consumer.value)})
            for path, raw in original.items():
                path.write_bytes(raw)
    finally:
        for path, raw in original.items():
            path.write_bytes(raw)
    verify_bundle(package)
    (root / "model-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_complete_model_executes_with_source_pinned_cfg_phases_on_serial_target(selected):
    import sys
    from test_spike_model import TOOLS as TARGET_TOOLS, Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, dtype, _, package = selected
    inputs, expected = invocation(root, dtype, 19, 8)
    reference = root / "target-reference.npz"
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        report = Q.qualify(package, root / "target", result_mode="execution-owned", comparison="words",
            inputs=inputs, reference=reference, merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"])
    finally:
        sys.path.remove(scripts)
    assert report["evaluation"]["result_count"] == 12
    words(root / "target/outputs.npz", expected)
    reader = "src/merlin/frontends/kernel_barrier_evidence.py"
    assert json.loads((root / "target/build.json").read_text())["merlin_sources"][reader] == _record(
        Path(os.environ["T2M_MERLIN_ROOT"]) / reader)
