"""Runtime source coordinates retain non-dense views and complete model state."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from t2m.allocation_bindings import PointerArgument, ScalarArgument, convert_allocations
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_MODEL2MLIR_PYTHON", "T2M_MERLIN_ROOT",
    "T2M_MERLIN_PYTHON", "T2M_NATIVE_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/model/Merlin tools required")

NAMES = ("root_rows", "root_cols", "rows", "cols", "s0", "s1", "origin", "steps", "bias")


def implementation(root, dtype="i64"):
    from t2m.examples.model_view_kernels import update_source_view

    root.mkdir(parents=True, exist_ok=False)
    source = root / "source.ttir"
    pointer = "*i64" if dtype == "i64" else "*fp32"
    source.write_text(capture_ttir(update_source_view, {
        **{name: pointer for name in ("owner", "view", "out")},
        **{name: "i32" for name in ("rows", "cols", "s0", "s1", "steps")},
        "bias": "i64", "BM": "constexpr", "BN": "constexpr",
    }, {"BM": 8, "BN": 4}))
    bounds = ((2, 9), (2, 17), (0, 17), (0, 17), (1, 64), (1, 64), (0, 153), (0, 3), (-9, 9))
    scalars = tuple(RuntimeScalar(name, "i64" if name in ("origin", "bias") else "i32", position + 2,
                                 bounds=bound) for position, (name, bound) in enumerate(zip(NAMES, bounds, strict=True)))
    abi = TensorABI("dynamic_source_view_root", (
        TensorBuffer("state", ("root_rows", "root_cols"), "readwrite", dtype),
        TensorBuffer("out", ("rows", "cols"), "write", dtype),
    ), runtime_scalars=scalars, grid=(GridExtent("rows", 8), GridExtent("cols", 4)))
    arguments = (PointerArgument("state"), PointerArgument("state", "origin"), PointerArgument("out"),
        *(ScalarArgument(name) for name in ("rows", "cols", "s0", "s1", "steps", "bias")))
    return convert_allocations(source, abi, arguments, root / "translation", source_entry="update_source_view",
        optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile="guarded-buffer", address_policy="source-words")


def bindings(result, *, runtime=False):
    cases = ({"s0": 2, "origin": 1, "steps": 2, "bias": 5},
             {"s1": 3, "origin": 2, "steps": 1, "bias": -3},
             {"rows": 0, "s0": 1, "origin": 0, "steps": 3, "bias": 2})
    return [KernelBinding("t2m_demo::dynamic_source_view_update", result, operands={"state": "owner"},
        scalar_arguments=case, runtime_arguments={name: name for name in case},
        runtime_operands={name: name for name in NAMES if name not in case},
        source_views={"view": {"root": "state", "source_arg": 1,
            "shape": ["rows", "cols"], "strides": ["s0", "s1"], "offset": "origin"}},
        **({"source_view_policy": "runtime-logical-coordinates-v1"} if runtime else {})) for case in cases]


@pytest.fixture(scope="module", params=("i64", "f32"))
def selected(tmp_path_factory, request):
    from t2m.model_bundle import write_model_bundle

    root = tmp_path_factory.mktemp("dynamic-source-view-" + request.param)
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_dynamic_source_view_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture),
        "--dtype", request.param], capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    result = implementation(root / "kernel", request.param)
    package = write_model_bundle(capture, bindings(result, runtime=True), root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, capture, package, request.param


def test_complete_dynamic_root_views_reach_the_independent_merlin_interface(selected):
    from t2m.bundle import verify_bundle

    _root, capture, package, _dtype = selected
    receipt = verify_bundle(package)
    for original, target in receipt["parent_artifact_bindings"].items():
        assert (capture / original).read_bytes() == (package / target).read_bytes()
    trace = json.loads((package / "kernel-bindings.json").read_text())
    assert trace["execution_view_binding_profile"] == "t2m.execution-checked-call-views.v3"
    assert len(trace["calls"]) == 3 and len(trace["kernels"]) == 1
    assert all(call["source_view_policy"] == "runtime-logical-coordinates-v1" for call in trace["calls"])
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(
        package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(actual.files) == len(expected.files) == 9
        for name in actual.files:
            assert (actual[name].shape, actual[name].dtype, actual[name].tobytes()) == (
                expected[name].shape, expected[name].dtype, expected[name].tobytes())


def test_runtime_source_coordinates_do_not_authorize_rehashed_view_claims(selected):
    from test_executed_view_models import check_forged_views

    root, _capture, package, _dtype = selected
    check_forged_views(package, root)


def test_independent_shapes_and_parameters_reuse_one_complete_library(selected):
    from t2m.bundle import _record
    from t2m.consumer import invoke_merlin_host

    root, _capture, package, dtype = selected
    library = _record(package / "consumer/host/model.so")
    observations = []
    for ordinal, shape in enumerate(((2, 2), (9, 17), (2, 3), (7, 2))):
        storage = np.int64 if dtype == "i64" else np.float32
        x = (np.arange(np.prod(shape)).reshape(shape) % 11).astype(storage)
        weight = np.asarray((1 << 56) + ordinal + 1 if dtype == "i64" else -ordinal - 2, storage)
        bias = np.asarray(-ordinal - 1, storage)
        initial = np.add(x, weight, dtype=storage)
        first = initial.copy()
        first.T[1::2, :] += 10
        second = first.copy()
        second[::2, 2::3] -= 3
        wanted = (np.add(second, bias, dtype=storage), first, first.T[1::2, :].copy(),
            second[::2, 2::3].copy(), first.copy(), np.empty((0, shape[0]), storage),
            initial.T[1::2, :].copy(), initial, x)
        arguments = root / f"arguments-{ordinal}.npz"
        np.savez(arguments, in0=weight, in1=bias, in2=x)
        np.savez(root / f"references-{ordinal}.npz", **{f"out{i}": value for i, value in enumerate(wanted)})
        actual, report = invoke_merlin_host(package / "consumer", arguments, root / f"invoke-{ordinal}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for observed, expected in zip(actual, wanted, strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (expected.shape, expected.dtype, expected.tobytes())
        observations.append({"shape": shape, "results": len(actual)})
    assert _record(package / "consumer/host/model.so") == library
    (root / "replays.json").write_text(json.dumps(observations, indent=2) + "\n")


def test_actual_target_preserves_dynamic_source_coordinates_and_all_snapshots(selected, tmp_path, monkeypatch):
    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, _capture, package, dtype = selected
    # Earlier replay inputs deliberately include an empty boundary slice and
    # an independent ragged matrix. Run the replay helper if selected alone.
    ordinal = 0 if dtype == "i64" else 1
    if not (root / f"arguments-{ordinal}.npz").exists():
        test_independent_shapes_and_parameters_reuse_one_complete_library(selected)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    report = Q.qualify(package, tmp_path / "target", result_mode="execution-owned", comparison="words",
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        inputs=root / f"arguments-{ordinal}.npz", reference=root / f"references-{ordinal}.npz")
    assert report["evaluation"]["result_count"] == 9
    assert Q.verify(tmp_path / "target", merlin_root=os.environ["T2M_MERLIN_ROOT"],
                    python=os.environ["T2M_MERLIN_PYTHON"]) == report
