"""Complete dependent model preserves every cross-output observation."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host, observe_merlin_host
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_MODEL2MLIR_PYTHON", "T2M_MERLIN_ROOT",
    "T2M_MERLIN_PYTHON", "T2M_NATIVE_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/model/Merlin tools required")


@pytest.fixture(scope="module")
def selected(tmp_path_factory):
    from t2m.examples.interleaved_tile_kernels import interleaved_partition

    root = tmp_path_factory.mktemp("interleaved-model")
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_interleaved_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture)],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    bindings = []
    for kind, mode in enumerate(("cross", "readonly")):
        directory = root / f"kernel-{kind}"
        directory.mkdir()
        source = directory / "source.ttir"
        source.write_text(capture_ttir(interleaved_partition,
            {**{name: "*i64" for name in ("x", "bias", "a", "b", "seen_a", "seen_b")},
             "n": "i64", "control": "i64", "MODE": "constexpr", "B": "constexpr"}, {"MODE": mode, "B": 8}))
        abi = TensorABI("interleaved_partition", (TensorBuffer("x", ("n",), "read", "i64"),
            TensorBuffer("bias", (1,), "read", "i64"),
            *(TensorBuffer(name, ("n",), "write", "i64") for name in ("a", "b", "seen_a", "seen_b"))),
            grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i64", 6, bounds=(0, 31)),
                                                       RuntimeScalar("control", "i64", 7, bounds=(0, 29))))
        result = convert_shared(source, abi, directory / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
        assert not any(op.name.startswith("memref.") for op in result.module.walk())
        bindings.append(KernelBinding("t2m_demo::interleaved_partition", result, results=("a", "b", "seen_a", "seen_b"),
            scalar_arguments={"kind": kind}, runtime_operands={"n": "n", "control": "control"}))
    package = write_model_bundle(capture, bindings, root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, capture, package


def test_every_original_result_and_parent_archive_survives(selected):
    _root, capture, package = selected
    receipt = verify_bundle(package)
    for source, destination in receipt["parent_artifact_bindings"].items():
        assert (capture / source).read_bytes() == (package / destination).read_bytes()
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(
            package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(expected.files) == len(actual.files) == 11
        for key in expected.files:
            assert (actual[key].shape, actual[key].dtype, actual[key].tobytes()) == (
                expected[key].shape, expected[key].dtype, expected[key].tobytes())


def invocation(root, package, index):
    from test_interleaved_tiles import reference

    n = (2, 7, 31)[index]
    x = np.arange(n, dtype=np.int64) * 37 + (1 << 56)
    x[1::2] *= -1
    x[-1] = (-(1 << 63), (1 << 63)-1, -1)[index]
    bias = np.asarray([(1 << 53) + index], np.int64)
    a, b, sa, sb = reference(x, bias, n - 2, "cross")
    shifted = np.add(a, b)
    c, d, sc, sd = reference(shifted, bias, n - 2, "readonly")
    expected = (c, d, a, b, sa, sb, sc, sd, x, shifted, bias)
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    values = {"x": x, "bias": bias}
    inputs, golden = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
    np.savez(inputs, **{f"in{i}": values[row.get("weight") or row["name"]] for i, row in enumerate(manifest.values())})
    np.savez(golden, **{f"out{i}": value for i, value in enumerate(expected)})
    return inputs, golden, expected


def test_one_library_retains_both_calls_all_observations_and_learned_state(selected):
    root, _capture, package = selected
    library = _record(package / "consumer/host/model.so")
    for index in range(3):
        inputs, _reference, expected = invocation(root, package, index)
        actual, report = invoke_merlin_host(package / "consumer", inputs, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def test_original_exported_domain_violation_still_aborts_without_outputs(selected):
    root, _capture, package = selected
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    values = {"x": np.arange(1, dtype=np.int64), "bias": np.asarray([3], np.int64)}
    inputs = root / "invalid-domain.npz"
    np.savez(inputs, **{f"in{i}": values[row.get("weight") or row["name"]] for i, row in enumerate(manifest.values())})
    out = root / "aborted-domain"
    report = observe_merlin_host(package / "consumer", inputs, out,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert report["returncode"] == -6 and not (out / "outputs.npz").exists()
    (root / "domain-abort.json").write_text(json.dumps(report, indent=2) + "\n")


def test_actual_serial_images_retain_all_eleven_original_results(selected, monkeypatch):
    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    root, _capture, package = selected
    for index in (0, 2):
        inputs, reference, _expected = invocation(root, package, index)
        out = root / f"target-{index}"
        report = Q.qualify(package, out, result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"], inputs=inputs, reference=reference)
        assert report["evaluation"]["result_count"] == 11
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
