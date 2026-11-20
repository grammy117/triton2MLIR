"""Complete dynamic PyTorch model retains source-indexed while recurrences."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
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
    from t2m.examples.while_index_kernels import prefix_while

    root = tmp_path_factory.mktemp("indexed-while-model")
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_indexed_while_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture)],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    source = root / "source.ttir"
    source.write_text(capture_ttir(prefix_while, {**{name: "*i64" for name in ("x", "state", "before", "after")}, "n": "i64"}, {}))
    abi = TensorABI("prefix_while", (TensorBuffer("x", ("n",), "read", "i64"),
        TensorBuffer("state", (1,), "readwrite", "i64"), TensorBuffer("before", (1,), "write", "i64"),
        TensorBuffer("after", (1,), "write", "i64")),
        runtime_scalars=(RuntimeScalar("n", "i64", 4, bounds=(0, 31)),))
    result = convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    binding = KernelBinding("t2m_demo::indexed_while", result, results=("before", "after", "state"),
                            runtime_operands={"n": "n"})
    package = write_model_bundle(capture, [binding], root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, capture, package


def test_original_model_results_and_parent_bytes_survive(selected):
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
    n, bias, initial = ((2, -7, (1 << 63) - 1), (31, 5, (1 << 55) + 1), (11, -9, -(1 << 63)))[index]
    x = np.arange(n, dtype=np.int64) * 37 + ((1 << 63) - 100)
    counter, bias = np.asarray([initial], np.int64), np.asarray(bias, np.int64)
    shifted = x + bias

    def fold(values, state):
        for i, value in enumerate(values):
            number = (int(value) + i + (1 << 63)) % (1 << 64) - (1 << 63)
            state ^= number
        return np.asarray([state], np.int64)

    middle = fold(x, initial)
    final = fold(shifted, int(middle[0]))
    expected = (counter.copy(), counter.copy(), middle, middle.copy(), middle.copy(), final,
                final.copy(), x, counter, bias, shifted)
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    values = {"x": x, "counter": counter, "bias": bias}
    arguments, references = root / f"arguments-{index}.npz", root / f"references-{index}.npz"
    np.savez(arguments, **{f"in{i}": values[row.get("weight") or row["name"]]
                         for i, row in enumerate(manifest.values())})
    np.savez(references, **{f"out{i}": value for i, value in enumerate(expected)})
    return arguments, references, expected


def test_one_library_preserves_changed_lengths_parameters_and_snapshots(selected):
    root, _capture, package = selected
    library = _record(package / "consumer/host/model.so")
    for index in range(3):
        arguments, _references, expected = invocation(root, package, index)
        actual, report = invoke_merlin_host(package / "consumer", arguments, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for observed, wanted in zip(actual, expected, strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def test_actual_serial_model_keeps_both_loops_and_every_result(selected, monkeypatch):
    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    root, _capture, package = selected
    for index in (0, 1):
        inputs, references, expected = invocation(root, package, index)
        out = root / f"target-{index}"
        report = Q.qualify(package, out, result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"], inputs=inputs, reference=references)
        assert report["evaluation"]["result_count"] == len(expected) == 11
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"],
                        python=os.environ["T2M_MERLIN_PYTHON"]) == report
