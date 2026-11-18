"""Real selected unit/scalar source views retain complete model snapshots."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_dynamic_source_view_models import NAMES, TOOLS, implementation

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle

pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/model/Merlin tools required")


def bindings(matrix):
    cases = (
        (matrix, {"rows": 1, "s1": 1, "steps": 2, "bias": 5}, False),
        (matrix, {"rows": 1, "steps": 1, "bias": -3}, False),
        (matrix, {"rows": 1, "cols": 1, "s0": 1, "s1": 1, "steps": 3, "bias": 2}, True),
        (matrix, {"rows": 1, "cols": 0, "s1": 1, "steps": 2, "bias": 4}, False),
    )
    return [KernelBinding("t2m_demo::rank_selected_scalar_update" if is_scalar else "t2m_demo::rank_selected_view_update",
        result, operands={"state": "owner"}, scalar_arguments=case, runtime_arguments={name: name for name in case},
        runtime_operands={name: name for name in NAMES if name not in case},
        tensor_views="dense-row-major-runtime-v1" if is_scalar else None,
        source_view_policy="runtime-logical-coordinates-v1", source_views={"view": {
            "root": "state", "source_arg": 1, "shape": [] if is_scalar else ["rows", "cols"],
            "strides": [] if is_scalar else ["s0", "s1"], "offset": "origin"}})
        for result, case, is_scalar in cases]


@pytest.fixture(scope="module", params=("i64", "f32"))
def selected(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("rank-selected-view-" + request.param)
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_rank_selected_view_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture),
        "--dtype", request.param], capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    shape = (1, "root_rows", "root_cols")
    matrix = implementation(root / "matrix", request.param, root_shape=shape, stride_bound=9 * 17)
    package = write_model_bundle(capture, bindings(matrix), root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, capture, package, request.param


def test_rank_reduction_keeps_every_original_result_and_pointer_root(selected):
    _root, capture, package, _dtype = selected
    receipt = verify_bundle(package)
    for source, target in receipt["parent_artifact_bindings"].items():
        assert (capture / source).read_bytes() == (package / target).read_bytes()
    trace = json.loads((package / "kernel-bindings.json").read_text())
    assert trace["execution_view_binding_profile"] == "t2m.execution-checked-call-views.v3"
    assert len(trace["calls"]) == 4 and len(trace["kernels"]) == 1
    scalar = trace["calls"][2]["source_tensor_views"]["view"]
    assert scalar["view_type"].startswith("tensor<") and scalar["binding"]["shape"] == []
    assert any(step.get("retained_axes") == [] for step in scalar["layout"]["steps"])
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(
        package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(expected.files) == len(actual.files) == 13
        for name in actual.files:
            assert (actual[name].shape, actual[name].dtype, actual[name].tobytes()) == (
                expected[name].shape, expected[name].dtype, expected[name].tobytes())


def invocation_files(root, dtype, ordinal):
    shape = ((1, 2, 2), (1, 9, 17), (1, 2, 3))[ordinal]
    storage = np.int64 if dtype == "i64" else np.float32
    x = (np.arange(np.prod(shape)).reshape(shape) % 11).astype(storage)
    weight = np.asarray((1 << 56) + ordinal + 1 if dtype == "i64" else -ordinal - 2, storage)
    bias = np.asarray(-ordinal - 1, storage)
    initial = np.add(x, weight, dtype=storage)
    first = initial.copy()
    first[:, 1, :] += 10
    second = first.copy()
    second[:, :, -1] -= 3
    third = second.copy()
    third[0, 1, -1] += 6
    wanted = (np.add(third, bias, dtype=storage), first, first[:, 1, :].copy(), second[:, :, -1].copy(),
        third, np.asarray(third[0, 1, -1]), third.copy(), np.empty((1, 0), storage),
        initial[:, 1, :].copy(), first[:, :, -1].copy(), np.asarray(second[0, 1, -1]), initial, x)
    arguments, references = root / f"arguments-{ordinal}.npz", root / f"references-{ordinal}.npz"
    np.savez(arguments, in0=weight, in1=bias, in2=x)
    np.savez(references, **{f"out{i}": value for i, value in enumerate(wanted)})
    return arguments, references, wanted


def test_dynamic_geometry_and_parameters_reuse_the_complete_library(selected):
    root, _capture, package, dtype = selected
    library = _record(package / "consumer/host/model.so")
    for ordinal in range(3):
        arguments, _reference, wanted = invocation_files(root, dtype, ordinal)
        actual, report = invoke_merlin_host(package / "consumer", arguments, root / f"invoke-{ordinal}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for observed, expected in zip(actual, wanted, strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (expected.shape, expected.dtype, expected.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def check_rehashed_rank_claims(package, root):
    from test_executed_model_packages import rehash

    from t2m.consumer import check_merlin_interface
    from t2m.executed_model_binding import serialize
    from t2m.shared_tensor import parse_shared

    paths = [package / name for name in ("capture_receipt.json", "kernel-bindings.json", "model.mlir")]
    original = {path: path.read_bytes() for path in paths}
    outcomes = []
    try:
        for damage in ("retained-axis", "scalar-rank", "coordinate-guard", "source-index-guard"):
            trace = json.loads(original[paths[1]])
            if damage in ("retained-axis", "scalar-rank"):
                call = trace["calls"][0 if damage == "retained-axis" else 2]
                step = next(step for step in call["source_tensor_views"]["view"]["layout"]["steps"]
                            if "retained_axes" in step and (damage != "scalar-rank" or step["retained_axes"] == []))
                step["retained_axes"] = [1, 2] if damage == "retained-axis" else [0]
            else:
                module = parse_shared(original[paths[2]].decode())
                guard = next(op for op in module.walk() if op.name == "cf.assert" and
                    ("kernel argument disagrees" in op.msg.data if damage == "coordinate-guard"
                     else "selection index is below selected dimension" in op.msg.data))
                guard.parent.erase_op(guard)
                (package / "model.mlir").write_text(serialize(module))
                trace["mlir_sha256"] = _record(package / "model.mlir")["sha256"]
            (package / "kernel-bindings.json").write_text(json.dumps(trace))
            rehash(package)
            with pytest.raises(ValueError) as producer:
                verify_bundle(package)
            output = root / ("refused-" + damage)
            with pytest.raises(ValueError) as consumer:
                check_merlin_interface(package, output, merlin_root=os.environ["T2M_MERLIN_ROOT"],
                    python=os.environ["T2M_MERLIN_PYTHON"], result_mode="execution-owned")
            assert not (output / "compiler").exists()
            outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": str(consumer.value)})
            for path, data in original.items():
                path.write_bytes(data)
    finally:
        for path, data in original.items():
            path.write_bytes(data)
    verify_bundle(package)
    (root / "rank-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_rehashed_rank_and_guard_claims_need_the_complete_source_expansion(selected):
    root, _capture, package, _dtype = selected
    check_rehashed_rank_claims(package, root)


def check_target(package, root, dtype):
    import sys

    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    arguments, references, _wanted = invocation_files(root, dtype, 1)
    directory = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, directory)
    try:
        report = Q.qualify(package, root / "target", result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
            inputs=arguments, reference=references)
        assert report["evaluation"]["result_count"] == 13
        assert Q.verify(root / "target", merlin_root=os.environ["T2M_MERLIN_ROOT"],
                        python=os.environ["T2M_MERLIN_PYTHON"]) == report
    finally:
        sys.path.remove(directory)


def test_actual_target_preserves_selected_unit_axes_scalars_and_every_snapshot(selected):
    root, _capture, package, dtype = selected
    check_target(package, root, dtype)
