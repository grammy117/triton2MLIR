"""Shared mutable views retain original storage across complete actual kernels."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from t2m.allocation_bindings import PointerArgument, ScalarArgument, convert_allocations
from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_MODEL2MLIR_PYTHON", "T2M_MERLIN_ROOT",
    "T2M_MERLIN_PYTHON", "T2M_NATIVE_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/model/Merlin tools required")


def implementation(root, kind, *, profile="auto"):
    from t2m.examples.mutating_views import update_matrix_views, update_views

    root.mkdir()
    source = root / "source.ttir"
    matrix_views = kind == "permuted-base"
    scalar_names = ("left_n", "right_n", "left_stride", "right_stride")
    if matrix_views:
        scalar_names += ("left_k", "right_k", "left_k_stride", "right_k_stride")
    kernel = update_matrix_views if matrix_views else update_views
    source.write_text(capture_ttir(kernel, {**{name: "*i64" for name in ("left", "right", "old_left", "old_right", "bias_ptr")},
        **{name: "i64" for name in scalar_names}}, {}))
    dimensions = (("root_a", "root_b", "root_c") if matrix_views else
                  ("root_rows", "root_cols") if kind in ("matrix", "transposed-base") else ("root_n",))
    fields = {name: f"@base0.size{axis}" for axis, name in enumerate(dimensions)}
    for index, name in enumerate(("left", "right")):
        fields.update({name + "_n": f"@view{index}.size0", name + "_stride": f"@view{index}.stride0",
                       name + "_offset": f"@view{index}.offset"})
        if matrix_views:
            fields.update({name + "_k": f"@view{index}.size1", name + "_k_stride": f"@view{index}.stride1"})
    scalars = tuple(RuntimeScalar(name, "i64", 4 + index, bounds=(0, 63)) for index, name in enumerate(fields))
    abi = TensorABI("mutable_view_storage", (
        TensorBuffer("bias", (1,), "read", "i64"), TensorBuffer("state", dimensions, "readwrite", "i64"),
        TensorBuffer("old_left", ("left_n", "left_k") if matrix_views else ("left_n",), "write", "i64"),
        TensorBuffer("old_right", ("right_n", "right_k") if matrix_views else ("right_n",), "write", "i64")),
        runtime_scalars=scalars)
    arguments = (PointerArgument("state", "left_offset"), PointerArgument("state", "right_offset"),
        PointerArgument("old_left"), PointerArgument("old_right"), PointerArgument("bias"),
        *(ScalarArgument(name) for name in scalar_names))
    converted = convert_allocations(source, abi, arguments, root / "translation", source_entry=kernel.__name__,
        optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile=profile)
    return KernelBinding("t2m_demo::mutating_views", converted, operands={"bias": "bias", "state": "@base0"},
        results=("old_left", "old_right", "state"), runtime_operands=fields, mutation_arguments={"left": 0, "right": 1},
        tensor_views="dense-row-major-runtime-v1")


@pytest.fixture(scope="module", params=("overlap", "stride", "same", "matrix", "transposed-base", "permuted-base"))
def selected(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("mutating-view-" + request.param)
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_mutating_view_model.py"
    done = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture),
        "--kind", request.param], capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert done.returncode == 0, done.stderr
    binding = implementation(root / "kernel", request.param)
    package = write_model_bundle(capture, [binding], root / "package", work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True, result_mode="execution-owned")
    return root, capture, package, request.param


def test_complete_shared_view_models_keep_one_allocation_and_all_snapshots(selected):
    _root, capture, package, kind = selected
    receipt = verify_bundle(package)
    for source, target in receipt["parent_artifact_bindings"].items():
        assert (capture / source).read_bytes() == (package / target).read_bytes()
    trace = json.loads((package / "kernel-bindings.json").read_text())
    assert trace["functional_kernel_binding_profile"] == "t2m.functional-kernel-binding.v2"
    assert len(trace["calls"]) == 2 and all(row["mutation_pointer_arguments"] == {"left": 0, "right": 1} for row in trace["calls"])
    kernel = next(iter(trace["kernels"].values()))
    assert kernel["memory_selection"]["attempts"][-1] == {
        "memory_profile": "guarded-buffer" if kind == "permuted-base" else "tensor",
        "address_policy": "source-words" if kind == "permuted-base" else "structured",
        "status": "admitted"}
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(
        package / "consumer/host/outputs.npz", allow_pickle=False) as actual:
        assert len(expected.files) == len(actual.files) == (8 if kind in ("transposed-base", "permuted-base") else 9)
        for key in expected.files:
            assert (actual[key].shape, actual[key].dtype, actual[key].tobytes()) == (
                expected[key].shape, expected[key].dtype, expected[key].tobytes())


def invocation_files(root, kind, index):
    shape = ((2, 2), (7, 9), (3, 5))[index] if kind in ("matrix", "transposed-base") else ((4,), (23,), (31,))[index]
    if kind == "permuted-base":
        shape = ((2, 2, 2), (4, 6, 7), (3, 5, 4))[index]
    x = np.arange(np.prod(shape), dtype=np.int64).reshape(shape) + (1 << 56)
    bias = np.asarray(index + 3, dtype=np.int64)
    current = x.T.copy(order="K") if kind == "transposed-base" else x.copy()
    if kind == "permuted-base":
        current = x.transpose(1, 2, 0).copy(order="K")
    before = current.copy()
    if kind == "overlap":
        left, right = current[1:-1], current[2:]
    elif kind == "stride":
        left, right = current[::2], current[1::2]
    elif kind in ("matrix", "transposed-base"):
        left, right = current[:, 1], current[:, -1]
    elif kind == "permuted-base":
        left, right = current[:, 1, :], current[:, -1, :]
    else:
        left, right = current[1:-1], current[1:-1]
    a, b = left.copy(), right.copy()
    left += bias
    right *= 3
    middle, c, d = current.copy(), left.copy(), right.copy()
    left += bias
    right *= 3
    wanted = (before, a, b, middle, c, d, current.copy(), bias)
    if kind not in ("transposed-base", "permuted-base"):
        wanted = (current.copy(), *wanted)
    arguments, references = root / f"arguments-{index}.npz", root / f"references-{index}.npz"
    np.savez(arguments, in0=bias, in1=x)
    np.savez(references, **{f"out{i}": value for i, value in enumerate(wanted)})
    return arguments, references, wanted


def test_changed_shapes_and_alias_patterns_reuse_the_original_library(selected):
    root, _capture, package, kind = selected
    library = _record(package / "consumer/host/model.so")
    for index in range(3):
        arguments, _references, wanted = invocation_files(root, kind, index)
        actual, report = invoke_merlin_host(package / "consumer", arguments, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for observed, expected in zip(actual, wanted, strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (expected.shape, expected.dtype, expected.tobytes())
    assert _record(package / "consumer/host/model.so") == library


def test_rehashed_view_pointer_and_profile_claims_refuse_both_readers(selected):
    from test_executed_model_packages import rehash

    from t2m.consumer import check_merlin_interface
    from t2m.executed_model_binding import serialize
    from t2m.shared_tensor import parse_shared

    root, _capture, package, _kind = selected
    paths = [package / name for name in ("capture_receipt.json", "kernel-bindings.json", "model.mlir")]
    original = {path: path.read_bytes() for path in paths}
    outcomes = []
    try:
        for damage in ("missing-profile", "downgraded-profile", "swapped-pointers", "missing-pointer", "coordinate-guard"):
            trace = json.loads(original[paths[1]])
            if damage == "missing-profile":
                trace.pop("functional_kernel_binding_profile")
            elif damage == "downgraded-profile":
                trace["functional_kernel_binding_profile"] = "t2m.functional-kernel-binding.v1"
            elif damage == "swapped-pointers":
                trace["calls"][0]["mutation_pointer_arguments"] = {"left": 1, "right": 0}
            elif damage == "missing-pointer":
                trace["calls"][0].pop("mutation_pointer_arguments")
            else:
                module = parse_shared(original[paths[2]].decode())
                guard = next(operation for operation in module.walk() if operation.name == "cf.assert"
                             and "mutable-view coordinate" in operation.msg.data)
                guard.parent.erase_op(guard)
                paths[2].write_text(serialize(module))
                trace["mlir_sha256"] = _record(paths[2])["sha256"]
            paths[1].write_text(json.dumps(trace))
            rehash(package)
            with pytest.raises(ValueError) as producer:
                verify_bundle(package)
            with pytest.raises(ValueError) as consumer:
                check_merlin_interface(package, root / ("refused-" + damage), merlin_root=os.environ["T2M_MERLIN_ROOT"],
                    python=os.environ["T2M_MERLIN_PYTHON"], result_mode="execution-owned")
            outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": str(consumer.value)})
    finally:
        for path, content in original.items():
            path.write_bytes(content)
    verify_bundle(package)
    (root / "mutation-view-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")


def test_actual_serial_target_preserves_shared_view_order_and_every_result(selected):
    import sys

    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, _capture, package, kind = selected
    arguments, references, _wanted = invocation_files(root, kind, 1)
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        report = Q.qualify(package, root / "target", result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
            inputs=arguments, reference=references)
        assert report["evaluation"]["result_count"] == len(_wanted)
        assert Q.verify(root / "target", merlin_root=os.environ["T2M_MERLIN_ROOT"],
                        python=os.environ["T2M_MERLIN_PYTHON"]) == report
    finally:
        sys.path.remove(scripts)


def test_original_public_shape_domain_still_aborts_before_results(selected):
    import signal

    from t2m.consumer import observe_merlin_host

    root, _capture, package, kind = selected
    shape = (2, 10) if kind in ("matrix", "transposed-base") else (3,)
    if kind == "permuted-base":
        shape = (5, 2, 2)
    arguments = root / "invalid-domain.npz"
    np.savez(arguments, in0=np.asarray(2, np.int64), in1=np.arange(np.prod(shape), dtype=np.int64).reshape(shape))
    destination = root / "aborted-domain"
    outcome = observe_merlin_host(package / "consumer", arguments, destination,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert outcome["returncode"] == -signal.SIGABRT
    assert not (destination / "outputs.npz").exists()
