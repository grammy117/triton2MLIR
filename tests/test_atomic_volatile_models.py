"""Mixed atomic/volatile source kernels retain complete original model state."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import test_atomic_phase_models as common
from test_atomic_volatile import TOOLS, implementation

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle


@pytest.fixture(scope="module", params=["i64", "f32"])
def selected(tmp_path_factory, request):
    if not TOOLS or not os.environ.get("T2M_MODEL2MLIR_PYTHON"):
        pytest.skip("existing source/model/native/Merlin tools required")
    root = tmp_path_factory.mktemp("atomic-volatile-model-" + request.param)
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_atomic_phase_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture), "--dtype", request.param,
        "--volatile"], capture_output=True, text=True, timeout=60,
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


def invocation(root, dtype, n, index):
    dt = np.int64 if dtype == "i64" else np.float32
    values = np.arange(n, dtype=dt) + index - 8
    state = np.arange(32, dtype=dt) - index
    if dtype == "i64":
        state += 1 << 55
    bias = np.asarray(index - 2, dtype=dt)
    inputs = root / f"inputs-{index}.npz"
    np.savez(inputs, in0=bias, in1=state, in2=values)
    return inputs, common.reference(values, state, bias)


def test_complete_model_retains_every_parameter_snapshot_and_mixed_effect_result(selected):
    common.test_complete_model_retains_parameters_snapshots_and_atomic_results(selected)


def test_changed_mixed_effect_model_shapes_and_parameters_reuse_the_original_library(selected):
    root, dtype, _, package = selected
    library = _record(package / "consumer/host/model.so")
    for index, n in enumerate((2, 17, 31)):
        inputs, expected = invocation(root, dtype, n, index)
        actual, observation = invoke_merlin_host(package / "consumer", inputs, root / f"replay-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert observation["functional_inputs_preserved"]
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def test_rehashed_effect_components_cannot_replace_either_original_model_contract(selected):
    from copy import deepcopy
    from test_executed_model_packages import rehash
    from t2m.consumer import check_merlin_interface
    from t2m.source_capsule import record

    root, _, _, package = selected
    paths = [package / name for name in ("capture_receipt.json", "kernel-bindings.json", "kernels/0/provenance.json",
        "kernels/0/allocation-protection.json", "kernels/0/memory-admission.json", "kernels/0/word-stage.json")]
    original = {path: path.read_bytes() for path in paths}
    outcomes = []
    try:
        for damage in ("atomic-selection", "volatile-selection", "atomic-profile", "volatile-profile", "lost-observation", "cleanup"):
            protection, admission, stage = (json.loads(original[paths[i]]) for i in (3, 4, 5))
            provenance = json.loads(original[paths[2]])
            if damage.endswith("selection"):
                protection["barrier_conversions"][0]["contract"].pop(damage.split("-")[0] + "_effects")
            elif damage == "atomic-profile":
                admission["effect_profile"] = provenance["memory"]["effect_profile"] = "ordinary"
            elif damage == "volatile-profile":
                admission["volatile_profile"] = provenance["memory"]["volatile_profile"] = "ordinary"
            elif damage == "lost-observation":
                admission["volatile_adaptations"].pop(1)
                provenance["memory"]["volatile_adaptations"] = admission["volatile_adaptations"]
            else:
                stage["compute"]["native_stage"]["command"].insert(1, "--canonicalize")
            for path, value in zip(paths[3:], (protection, admission, stage), strict=True):
                path.write_text(json.dumps(value))
            provenance["source_argument_binding"]["source_capsule"]["artifacts"]["allocation-protection.json"] = record(paths[3].read_bytes())
            provenance["memory"]["admission_sha256"] = record(paths[4].read_bytes())["sha256"]
            provenance["memory"]["word_stage_sha256"] = record(paths[5].read_bytes())["sha256"]
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
            outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": str(consumer.value)})
            for path, contents in original.items():
                path.write_bytes(contents)
    finally:
        for path, contents in original.items():
            path.write_bytes(contents)
    verify_bundle(package)
    (root / "model-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_original_mixed_effect_model_domain_guard_still_aborts(selected):
    import signal
    from t2m.consumer import observe_merlin_host

    root, dtype, _, package = selected
    inputs, _ = invocation(root, dtype, 1, 50)
    work = root / "domain-abort"
    outcome = observe_merlin_host(package / "consumer", inputs, work,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert outcome["returncode"] == -signal.SIGABRT and not outcome["timed_out"]
    assert not (work / "outputs.npz").exists()


def test_actual_serial_target_preserves_the_whole_mixed_effect_model(selected):
    import sys
    from test_spike_model import TOOLS as TARGET_TOOLS, Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, dtype, _, package = selected
    inputs, expected = invocation(root, dtype, 31, 8)
    reference = root / "target-reference.npz"
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        report = Q.qualify(package, root / "target", result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"], inputs=inputs, reference=reference)
        assert report["evaluation"]["result_count"] == 13
    finally:
        sys.path.remove(scripts)


def test_public_binding_cli_preserves_global_and_per_kernel_atomic_numeric_policies(selected):
    from t2m.cli import main

    root, dtype, capture, package = selected
    if dtype != "f32":
        pytest.skip("f32 atomic policy transport belongs to the floating model")
    spec = json.loads((package / "kernels/0/spec.json").read_text())
    allocation = json.loads((package / "kernels/0/allocation-arguments.json").read_text())
    for case, global_policy, override in (("global", "ieee", None), ("override", "ftz", "ieee")):
        kernels = []
        for steps, empty in ((2, False), (3, False), (1, True)):
            kernels.append({"kernel": "t2m_demo::atomic_phases", "ttir": str(root / "kernel/source.ttir"),
                "abi": spec, "allocation_bindings": allocation, "operands": {"x": "x", "state": "state"},
                "results": ["state", "old", "seen"],
                "scalar_arguments": {"steps": steps, **({"n": 0} if empty else {})},
                "runtime_arguments": {"steps": "steps", **({"n": "n"} if empty else {})},
                "runtime_operands": {} if empty else {"n": "n"},
                **({"f32_atomic_add": override} if override is not None else {})})
        configuration = root / ("cli-" + case + ".json")
        configuration.write_text(json.dumps({"schema": "t2m.kernel-bindings.v1", "kernels": kernels}))
        output = root / ("cli-" + case)
        assert main(["bind-model", str(capture), "--bindings", str(configuration), "--out", str(output),
            "--optimizer", os.environ["TRITON_SHARED_OPT"], "--memory-profile", "auto", "--effect-profile", "atomic-volatile",
            "--f32-atomic-add", global_policy, "--python", os.environ["T2M_NATIVE_PYTHON"],
            "--merlin-root", os.environ["T2M_MERLIN_ROOT"], "--merlin-python", os.environ["T2M_MERLIN_PYTHON"],
            "--merlin-clang", os.environ["T2M_MERLIN_CLANG"], "--merlin-host", "--result-mode", "execution-owned"]) == 0
        verify_bundle(output)
        evidence = json.loads((output / "kernels/0/provenance.json").read_text())
        assert evidence["memory"]["f32_atomic_add"] == "ieee"
        with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(
            output / "consumer/host/outputs.npz", allow_pickle=False) as actual:
            assert len(actual.files) == len(expected.files) == 13
            for name in expected.files:
                a, b = actual[name], expected[name]
                assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
