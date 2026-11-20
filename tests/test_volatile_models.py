"""Original volatile kernels remain executable through complete model packages."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


def implementation(root, dtype):
    from t2m.allocation_bindings import PointerArgument, ScalarArgument, convert_allocations
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.volatile_kernels import volatile_updates
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer

    source = root / "source.ttir"
    source_type = "fp32" if dtype == "f32" else dtype
    source.write_text(capture_ttir(volatile_updates,
        {**{name: "*" + source_type for name in ("x", "state", "old", "seen")},
         "n": "i32", "steps": "i32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("allocation_volatile_updates", (TensorBuffer("x", ("n",), dtype=dtype),
        TensorBuffer("state", ("n",), "readwrite", dtype), TensorBuffer("old", ("n",), "write", dtype),
        TensorBuffer("seen", ("n",), "write", dtype)), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)), RuntimeScalar("steps", "i32", 5, bounds=(0, 3))))
    return convert_allocations(source, abi,
        (*[PointerArgument(name) for name in ("x", "state", "old", "seen")], ScalarArgument("n"), ScalarArgument("steps")),
        root / "translation", source_entry="volatile_updates", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile="volatile")


@pytest.fixture(scope="module", params=["i64", "f32"])
def selected(tmp_path_factory, request):
    if not TOOLS:
        pytest.skip("existing source/model/native/Merlin tools required")
    root = tmp_path_factory.mktemp("volatile-model-" + request.param)
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_volatile_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture), "--dtype", request.param],
        capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    work = root / "kernel"
    work.mkdir()
    kernel = implementation(work, request.param)
    bindings = [KernelBinding("t2m_demo::volatile_updates", kernel, operands={"x": "x", "state": "state"},
        results=("state", "old", "seen"), scalar_arguments={"steps": steps, **({"n": 0} if empty else {})},
        runtime_arguments={"steps": "steps", **({"n": "n"} if empty else {})},
        runtime_operands={} if empty else {"n": "n"}) for steps, empty in ((2, False), (3, False), (1, True))]
    package = write_model_bundle(capture, bindings, root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, request.param, capture, package


def test_complete_model_retains_every_snapshot_parameter_and_volatile_result(selected):
    _, _, capture, package = selected
    receipt = verify_bundle(package)
    for name, destination in receipt["parent_artifact_bindings"].items():
        assert (capture / name).read_bytes() == (package / destination).read_bytes()
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(
        package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(actual.files) == len(expected.files) == 14
        for name in expected.files:
            a, b = actual[name], expected[name]
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
    calls = json.loads((package / "kernel-bindings.json").read_text())
    assert len(calls["calls"]) == 3 and len(calls["kernels"]) == 1


def reference(x, bias, initial_word):
    original, initial = x + bias, x + initial_word
    first, old1 = initial + original + original, initial + original
    old2 = first + original + original
    second = old2 + original
    empty = np.empty(0, dtype=x.dtype)
    return [second, first, old1, first.copy(), old2, second.copy(), original, x, initial,
                empty, empty.copy(), empty.copy(), bias, initial_word]


def invocation(root, dtype, n, index):
    dt = np.int64 if dtype == "i64" else np.float32
    x = np.arange(n, dtype=dt) - 5 + index
    bias, initial_word = np.asarray(index - 2, dtype=dt), np.asarray(7 - index, dtype=dt)
    expected = reference(x, bias, initial_word)
    inputs = root / f"inputs-{index}.npz"
    np.savez(inputs, in0=bias, in1=initial_word, in2=x)
    return inputs, expected


def test_runtime_shapes_and_parameters_reuse_the_original_volatile_model_library(selected):
    root, dtype, _, package = selected
    before = _record(package / "consumer/host/model.so")
    for index, n in enumerate([2, 17, 31]):
        inputs, expected = invocation(root, dtype, n, index)
        actual, observation = invoke_merlin_host(package / "consumer", inputs, root / f"replay-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert observation["functional_inputs_preserved"]
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
    assert _record(package / "consumer/host/model.so") == before


def test_actual_serial_target_retains_the_whole_volatile_model(selected):
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
        assert report["evaluation"]["result_count"] == 14
    finally:
        sys.path.remove(scripts)


def test_rehashed_volatile_source_and_effect_claims_refuse_both_model_readers(selected):
    from copy import deepcopy
    from test_executed_model_packages import rehash
    from t2m.consumer import check_merlin_interface
    from t2m.source_capsule import record

    root, _, _, package = selected
    paths = [package / name for name in ("capture_receipt.json", "kernel-bindings.json", "kernels/0/provenance.json",
        "kernels/0/memory-admission.json", "kernels/0/word-stage.json")]
    original = {path: path.read_bytes() for path in paths}
    outcomes = []
    try:
        for damage in ("profile", "lost-observation", "cleanup"):
            provenance = json.loads(original[paths[2]])
            admission = json.loads(original[paths[3]])
            stage = json.loads(original[paths[4]])
            if damage == "profile":
                provenance["memory"]["effect_profile"] = "ordinary"
            elif damage == "lost-observation":
                admission["volatile_adaptations"].pop(1)
                provenance["memory"]["volatile_adaptations"] = admission["volatile_adaptations"]
            else:
                stage["compute"]["native_stage"]["command"].insert(1, "--canonicalize")
            paths[3].write_text(json.dumps(admission))
            paths[4].write_text(json.dumps(stage))
            provenance["memory"]["admission_sha256"] = record(paths[3].read_bytes())["sha256"]
            provenance["memory"]["word_stage_sha256"] = record(paths[4].read_bytes())["sha256"]
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
            assert any(word in str(producer.value) for word in ("volatile", "source-word"))
            assert any(word in str(consumer.value) for word in ("volatile", "source-word"))
            outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": str(consumer.value)})
            for path, contents in original.items():
                path.write_bytes(contents)
    finally:
        for path, contents in original.items():
            path.write_bytes(contents)
    verify_bundle(package)
    (root / "model-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_original_volatile_model_domain_guard_still_aborts(selected):
    import signal
    from t2m.consumer import observe_merlin_host

    root, dtype, _, package = selected
    inputs, _ = invocation(root, dtype, 1, 50)
    work = root / "domain-abort"
    outcome = observe_merlin_host(package / "consumer", inputs, work,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert outcome["returncode"] == -signal.SIGABRT and not outcome["timed_out"]
    assert not (work / "outputs.npz").exists()
