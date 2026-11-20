"""Complete dynamic models retain nested-loop algorithms and every result."""

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
    from t2m.examples.nested_overwrite_kernels import rectangular_transform

    root = tmp_path_factory.mktemp("rectangular-model")
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_rectangular_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture)],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    bindings = []
    for order in (0, 1):
        directory = root / f"kernel-{order}"
        directory.mkdir()
        source = directory / "source.ttir"
        source.write_text(capture_ttir(rectangular_transform, {**{name: "*i64" for name in ("x", "bias", "out", "seen")},
            "rows": "i64", "cols": "i64", "ORDER": "constexpr"}, {"ORDER": order}))
        abi = TensorABI("rectangular_transform", (TensorBuffer("x", ("rows", "cols"), "read", "i64"),
            TensorBuffer("bias", ("rows" if order == 0 else "cols",), "read", "i64"),
            TensorBuffer("out", ("rows", "cols"), "write", "i64"), TensorBuffer("seen", (2,), "write", "i64")),
            runtime_scalars=(RuntimeScalar("rows", "i64", 4, bounds=(0, 7)), RuntimeScalar("cols", "i64", 5, bounds=(0, 9))))
        result = convert_shared(source, abi, directory / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
        assert not any(op.name.startswith("memref.") for op in result.module.walk())
        bindings.append(KernelBinding("t2m_demo::rectangular_transform", result, results=("out", "seen"),
            scalar_arguments={"order": order}, runtime_operands={"rows": "rows", "cols": "cols"}))
    package = write_model_bundle(capture, bindings, root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, capture, package


def test_original_model_results_and_parent_archives_survive(selected):
    _root, capture, package = selected
    receipt = verify_bundle(package)
    for source, destination in receipt["parent_artifact_bindings"].items():
        assert (capture / source).read_bytes() == (package / destination).read_bytes()
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(
            package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(expected.files) == len(actual.files) == 7
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
    expected = (second, first, first.ravel()[[0, -1]], second.ravel()[[0, -1]], x, row_bias, col_bias)
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    values = {"x": x, "row_bias": row_bias, "col_bias": col_bias}
    arguments, reference = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
    np.savez(arguments, **{f"in{i}": values[row.get("weight") or row["name"]]
                         for i, row in enumerate(manifest.values())})
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
    return arguments, reference, expected


def test_one_library_preserves_independent_shapes_bias_values_and_both_traversals(selected):
    root, _capture, package = selected
    library = _record(package / "consumer/host/model.so")
    for index in range(3):
        arguments, _reference, expected = invocation(root, package, index)
        actual, report = invoke_merlin_host(package / "consumer", arguments, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for observed, wanted in zip(actual, expected, strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def test_original_model_shape_domain_still_aborts_before_any_result(selected):
    from t2m.consumer import observe_merlin_host

    root, _capture, package = selected
    manifest = json.loads((package / "weights.safetensors.manifest.json").read_text())
    outcomes = []
    for index, (rows, cols) in enumerate(((1, 6), (4, 1))):
        # These dimensions are safe for both original kernels but outside the
        # actual exported model domain. Kernel coverage cannot widen it.
        values = {"x": np.arange(rows * cols, dtype=np.int64).reshape(rows, cols),
                  "row_bias": np.arange(7, dtype=np.int64), "col_bias": np.arange(9, dtype=np.int64)}
        arguments = root / f"invalid-domain-{index}.npz"
        np.savez(arguments, **{f"in{i}": values[row.get("weight") or row["name"]]
                             for i, row in enumerate(manifest.values())})
        destination = root / f"aborted-domain-{index}"
        outcome = observe_merlin_host(package / "consumer", arguments, destination,
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert outcome["returncode"] == -6 and not (destination / "outputs.npz").exists()
        outcomes.append(outcome)
    (root / "domain-aborts.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_actual_serial_model_retains_all_results_of_both_source_loop_nests(selected, monkeypatch):
    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    root, _capture, package = selected
    for index in (0, 2):
        inputs, reference, expected = invocation(root, package, index)
        out = root / f"target-{index}"
        report = Q.qualify(package, out, result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"], inputs=inputs, reference=reference)
        assert report["evaluation"]["result_count"] == len(expected) == 7
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
