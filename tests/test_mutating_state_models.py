"""Actual mutating Triton models retain initial values and explicit state ports."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_MODEL2MLIR_PYTHON", "T2M_MERLIN_ROOT",
    "T2M_MERLIN_PYTHON", "T2M_NATIVE_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/model/Merlin tools required")


def implementation(root):
    from t2m.examples.mutating_state import observe_and_bump

    root.mkdir()
    source = root / "source.ttir"
    source.write_text(capture_ttir(observe_and_bump, {"state": "*i64", "observed": "*i64",
        "n": "i64", "bias": "i64", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("observe_and_bump", (TensorBuffer("state", ("n",), "readwrite", "i64"),
        TensorBuffer("observed", ("n",), "write", "i64")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i64", 2, bounds=(0, 31)), RuntimeScalar("bias", "i64", 3, bounds=(-9, 9))))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


@pytest.fixture(scope="module", params=("internal", "public", "buffer"))
def selected(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("mutating-state-" + request.param)
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_mutating_state_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture),
        "--kind", request.param], capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    result = implementation(root / "kernel")
    dynamic = request.param != "buffer"
    bindings = [KernelBinding("t2m_demo::mutating_state", result, results=("observed", "state"),
        scalar_arguments={"bias": bias, **({} if dynamic else {"n": 17})},
        runtime_arguments={"bias": "bias", **({} if dynamic else {"n": "n"})},
        runtime_operands={"n": "n"} if dynamic else {}) for bias in (3, -2)]
    package = write_model_bundle(capture, bindings, root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, capture, package, request.param


def test_every_state_and_user_result_matches_original_pytorch(selected):
    _root, capture, package, kind = selected
    receipt = verify_bundle(package)
    for source, target in receipt["parent_artifact_bindings"].items():
        assert (capture / source).read_bytes() == (package / target).read_bytes()
    trace = json.loads((package / "kernel-bindings.json").read_text())
    assert trace["functional_kernel_binding_profile"] == "t2m.functional-kernel-binding.v1"
    assert len(trace["calls"]) == 2
    assert all(row["source_result_order"] == ["observed", "state"] for row in trace["calls"])
    with np.load(capture / "inputs.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["in0"], np.arange(17, dtype=np.int64))
    with np.load(capture / "extra.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["buf::state"], np.arange(17, dtype=np.int64) + 10)
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(
        package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(expected.files) == len(actual.files) == (6 if kind == "internal" else 7)
        for key in expected.files:
            assert (actual[key].shape, actual[key].dtype, actual[key].tobytes()) == (
                expected[key].shape, expected[key].dtype, expected[key].tobytes())


def invocation_files(root, package, kind, index):
    n = (2, 23, 31)[index] if kind != "buffer" else 17
    x = (np.arange(n, dtype=np.int64) * (index + 3)) + (1 << 56)
    weight = np.asarray(-index - 5, np.int64)
    state = np.arange(17, dtype=np.int64) + 40 + index
    initial = state if kind == "buffer" else x if kind == "public" else x + weight
    user = (initial.copy(), initial.copy(), initial + 3, initial + 1, initial + 1 if kind == "public" else x, weight)
    wanted = user if kind == "internal" else (initial + 1, *user)
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    values = {"weight": weight, "state": state, "x": x}
    arguments, references = root / f"arguments-{index}.npz", root / f"references-{index}.npz"
    np.savez(arguments, **{f"in{i}": values[row.get("weight") or row["name"]]
                         for i, row in enumerate(manifest.values())})
    np.savez(references, **{f"out{i}": value for i, value in enumerate(wanted)})
    return arguments, references, wanted


def test_changed_inputs_reuse_library_and_preserve_borrowed_inputs(selected):
    root, _capture, package, kind = selected
    library = _record(package / "consumer/host/model.so")
    for index in range(3):
        arguments, _references, wanted = invocation_files(root, package, kind, index)
        actual, report = invoke_merlin_host(package / "consumer", arguments, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for observed, expected in zip(actual, wanted, strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (expected.shape, expected.dtype, expected.tobytes())
    assert _record(package / "consumer/host/model.so") == library
    if kind == "public":
        import signal

        from t2m.consumer import observe_merlin_host

        for n in (1, 32):
            arguments, _references, _wanted = invocation_files(root, package, kind, 0)
            with np.load(arguments, allow_pickle=False) as data:
                values = {key: data[key].copy() for key in data.files}
            values["in2"] = np.arange(n, dtype=np.int64)
            invalid = root / f"invalid-{n}.npz"
            np.savez(invalid, **values)
            destination = root / f"aborted-{n}"
            outcome = observe_merlin_host(package / "consumer", invalid, destination,
                merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
            assert outcome["returncode"] == -signal.SIGABRT
            assert not (destination / "outputs.npz").exists()


def test_rehashed_mutation_claims_refuse_in_both_readers(selected):
    from test_executed_model_packages import rehash

    from t2m.consumer import check_merlin_interface

    root, _capture, package, _kind = selected
    paths = [package / name for name in ("capture_receipt.json", "kernel-bindings.json")]
    original = {path: path.read_bytes() for path in paths}
    outcomes = []
    try:
        for damage in ("missing-profile", "unknown-profile", "fresh-output", "missing-base"):
            trace = json.loads(original[paths[1]])
            if damage == "missing-profile":
                trace.pop("functional_kernel_binding_profile")
            elif damage == "unknown-profile":
                trace["functional_kernel_binding_profile"] = "unchecked"
            elif damage == "fresh-output":
                trace["calls"][0]["source_result_order"] = ["state", "observed"]
            else:
                trace["calls"][0]["source_operand_bindings"] = {}
            paths[1].write_text(json.dumps(trace))
            rehash(package)
            with pytest.raises(ValueError) as producer:
                verify_bundle(package)
            with pytest.raises(ValueError) as consumer:
                check_merlin_interface(package, root / ("refused-" + damage),
                    merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
                    result_mode="execution-owned")
            outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": str(consumer.value)})
    finally:
        for path, content in original.items():
            path.write_bytes(content)
    verify_bundle(package)
    (root / "mutation-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_actual_serial_target_retains_every_functional_state_and_observation(selected):
    import sys

    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, _capture, package, kind = selected
    arguments, references, _wanted = invocation_files(root, package, kind, 1)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        report = Q.qualify(package, root / "target", result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
            inputs=arguments, reference=references)
        assert report["evaluation"]["result_count"] == (6 if kind == "internal" else 7)
        assert Q.verify(root / "target", merlin_root=os.environ["T2M_MERLIN_ROOT"],
                        python=os.environ["T2M_MERLIN_PYTHON"]) == report
    finally:
        sys.path.pop(0)
