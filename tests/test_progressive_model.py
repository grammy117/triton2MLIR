"""A complete model retains partial observations through both original calls."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host, observe_merlin_host
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
    from t2m.examples.partial_memory_kernels import progressive_transform

    root = tmp_path_factory.mktemp("progressive-model")
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_progressive_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture)],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    bindings = []
    for order in (0, 1):
        directory = root / f"kernel-{order}"
        directory.mkdir()
        source = directory / "source.ttir"
        source.write_text(capture_ttir(progressive_transform,
            {**{name: "*i64" for name in ("x", "bias", "out", "first", "last")},
             "rows": "i64", "cols": "i64", "ORDER": "constexpr", "MODE": "constexpr"}, {"ORDER": order, "MODE": "reread"}))
        axis = "rows" if order == 0 else "cols"
        abi = TensorABI("progressive_transform", (TensorBuffer("x", ("rows", "cols"), "read", "i64"),
            TensorBuffer("bias", (axis,), "read", "i64"), TensorBuffer("out", ("rows", "cols"), "write", "i64"),
            TensorBuffer("first", (axis,), "write", "i64"), TensorBuffer("last", (axis,), "write", "i64")),
            runtime_scalars=(RuntimeScalar("rows", "i64", 5, bounds=(0, 7)), RuntimeScalar("cols", "i64", 6, bounds=(0, 9))))
        result = convert_shared(source, abi, directory / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
        assert not any(op.name.startswith("memref.") for op in result.module.walk())
        bindings.append(KernelBinding("t2m_demo::progressive_transform", result, results=("out", "first", "last"),
            scalar_arguments={"order": order}, runtime_operands={"rows": "rows", "cols": "cols"}))
    package = write_model_bundle(capture, bindings, root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, capture, package


def test_complete_original_model_results_and_parent_archives_survive(selected):
    _root, capture, package = selected
    receipt = verify_bundle(package)
    for source, destination in receipt["parent_artifact_bindings"].items():
        assert (capture / source).read_bytes() == (package / destination).read_bytes()
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(
            package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(expected.files) == len(actual.files) == 9
        for key in expected.files:
            assert (actual[key].shape, actual[key].dtype, actual[key].tobytes()) == (
                expected[key].shape, expected[key].dtype, expected[key].tobytes())


def invocation(root, package, index):
    rows, cols = ((2, 9), (7, 2), (7, 9))[index]
    x = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols) * 37 + (1 << 56)
    row_bias = np.arange(7, dtype=np.int64) * 23 + index + (1 << 53)
    col_bias = np.arange(9, dtype=np.int64) * 41 - index + (1 << 54)
    first = x ^ row_bias[:rows, None]
    second = first ^ col_bias[None, :cols]
    expected = (second, first, np.bitwise_xor.reduce(first, axis=1), first[:, -1],
                np.bitwise_xor.reduce(second, axis=0), second[-1, :], x, row_bias, col_bias)
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    values = {"x": x, "row_bias": row_bias, "col_bias": col_bias}
    inputs, reference = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
    np.savez(inputs, **{f"in{i}": values[row.get("weight") or row["name"]] for i, row in enumerate(manifest.values())})
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
    return inputs, reference, expected


def test_one_library_preserves_original_recurrences_shapes_and_parameter_values(selected):
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


def test_original_public_shape_domain_aborts_without_any_result(selected):
    root, _capture, package = selected
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    values = {"x": np.arange(6, dtype=np.int64).reshape(1, 6), "row_bias": np.arange(7, dtype=np.int64),
              "col_bias": np.arange(9, dtype=np.int64)}
    inputs = root / "invalid-domain.npz"
    np.savez(inputs, **{f"in{i}": values[row.get("weight") or row["name"]] for i, row in enumerate(manifest.values())})
    out = root / "aborted-domain"
    report = observe_merlin_host(package / "consumer", inputs, out,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert report["returncode"] == -6 and not (out / "outputs.npz").exists()
    (root / "domain-abort.json").write_text(json.dumps(report, indent=2) + "\n")


def test_actual_serial_images_preserve_every_original_partial_observation(selected, monkeypatch):
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
        assert report["evaluation"]["result_count"] == 9
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
