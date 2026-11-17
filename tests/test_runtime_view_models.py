"""Real runtime matrix views retain count, parallel packing and dependent calls."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_compact_model import REQUIRED, pack_implementation
from test_kernel_size_binding import implementations
from test_spike_model import TOOLS as TARGET_TOOLS
from test_spike_model import Q

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.model_tensor_views import RUNTIME_POLICY

pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing source/model/Merlin toolchains required"
)


@pytest.fixture(scope="module")
def selected(tmp_path_factory):
    root = tmp_path_factory.mktemp("runtime-view-model")
    capture = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_compact_model.py"
    done = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--matrix-views",
            "--out",
            str(capture),
        ],
        capture_output=True,
        timeout=60,
        check=False,
        env=dict(
            os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)
        ),
    )
    assert done.returncode == 0, done.stderr.decode()
    ordinary = implementations(root / "ordinary", "auto")
    pack = pack_implementation(root / "pack")
    bindings = [
        KernelBinding(
            "t2m_compact_demo::" + name,
            implementation,
            tensor_views=RUNTIME_POLICY,
            runtime_operands=operands,
        )
        for name, implementation, operands in (
            ("positive_count", ordinary[0].implementation, {"capacity": "capacity"}),
            ("compact", pack, {"capacity": "capacity", "n": "n"}),
            ("prefix", ordinary[1].implementation, {"capacity": "capacity", "n": "n"}),
        )
    ]
    package = write_model_bundle(
        capture,
        bindings,
        root / "package",
        work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True,
        result_mode="execution-owned",
    )
    return root, capture, package


def test_original_dynamic_matrix_model_reaches_the_owned_merlin_interface(selected):
    _root, capture, package = selected
    receipt = verify_bundle(package)
    assert receipt["schema"] == "t2m.bound-model-receipt.v2"
    for original, destination in receipt["parent_artifact_bindings"].items():
        assert (capture / original).read_bytes() == (package / destination).read_bytes()
    trace = json.loads((package / "kernel-bindings.json").read_text())
    assert (
        trace["execution_view_binding_profile"] == "t2m.execution-checked-call-views.v2"
    )
    assert len(trace["calls"]) == len(trace["kernels"]) == 3
    signature = json.loads((package / "consumer/compiler/signature.json").read_text())
    assert signature["inputs"][1]["shape"] == [None, None]
    assert signature["outputs"][0]["shape"] == [1, None]
    with (
        np.load(capture / "goldens.npz", allow_pickle=False) as reference,
        np.load(package / "consumer/host/outputs.npz", allow_pickle=False) as actual,
    ):
        assert len(actual.files) == len(reference.files) == 7
        for name in actual.files:
            assert (actual[name].shape, actual[name].dtype, actual[name].tobytes()) == (
                reference[name].shape,
                reference[name].dtype,
                reference[name].tobytes(),
            )


DOMAINS = (((2, 2), -100), ((4, 7), 100), ((3, 5), -3), ((2, 7), 1))


def invocation_files(root, ordinal):
    shape, weight = DOMAINS[ordinal]
    x = np.resize(np.asarray([0, 1, -1, -(1 << 31), (1 << 31) - 1, 7], np.int32), shape)
    if ordinal < 2:
        x = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    parameter = np.asarray(weight, np.int32)
    original = np.add(x, parameter, dtype=np.int32)
    packed = original[original > 0].reshape(1, -1)
    wanted = (
        packed,
        packed,
        original,
        x,
        np.asarray([packed.size], np.int64),
        parameter,
        packed[:, ::2],
    )
    arguments, reference = (
        root / f"arguments-{ordinal}.npz",
        root / f"references-{ordinal}.npz",
    )
    np.savez(arguments, in0=parameter, in1=x)
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(wanted)})
    return arguments, reference, wanted


def test_one_library_preserves_independent_matrix_shapes_and_runtime_counts(selected):
    root, _capture, package = selected
    library = _record(package / "consumer/host/model.so")
    program = _record(package / "consumer/compiler/program.mlir")
    observations = []
    for ordinal, (shape, _weight) in enumerate(DOMAINS):
        arguments, _reference, wanted = invocation_files(root, ordinal)
        actual, report = invoke_merlin_host(
            package / "consumer",
            arguments,
            root / f"invoke-{ordinal}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert report["functional_inputs_preserved"]
        for value, expected in zip(actual, wanted, strict=True):
            assert (value.shape, value.dtype, value.tobytes()) == (
                expected.shape,
                expected.dtype,
                expected.tobytes(),
            )
        observations.append({"shape": list(shape), "selected": wanted[0].size})
    assert _record(package / "consumer/host/model.so") == library
    assert _record(package / "consumer/compiler/program.mlir") == program
    assert [row["selected"] for row in observations[:2]] == [0, 28]
    (root / "replays.json").write_text(json.dumps(observations, indent=2) + "\n")


@pytest.mark.parametrize("invocation", (None, 0, 1))
def test_actual_target_retains_runtime_matrix_views_and_all_original_results(
    selected, tmp_path, monkeypatch, invocation
):
    if not TARGET_TOOLS:
        pytest.skip("existing serial target tools required")
    root, _capture, package = selected
    if invocation is not None:
        invocation_files(root, invocation)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    report = Q.qualify(
        package,
        tmp_path / "target",
        result_mode="execution-owned",
        comparison="words",
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        **(
            {
                "inputs": root / f"arguments-{invocation}.npz",
                "reference": root / f"references-{invocation}.npz",
            }
            if invocation is not None
            else {}
        ),
    )
    assert report["evaluation"]["result_count"] == 7
    assert (
        Q.verify(
            tmp_path / "target",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        == report
    )


@pytest.mark.parametrize(
    "damage", ("profile", "policy", "overflow-guard", "dimension-operand")
)
def test_rehashed_runtime_view_claims_need_the_complete_original_expansion(
    selected, tmp_path, damage
):
    from test_executed_model_packages import rehash

    from t2m.consumer import check_merlin_interface
    from t2m.executed_model_binding import serialize
    from t2m.shared_tensor import parse_shared

    _root, _capture, package = selected
    paths = [
        package / name
        for name in ("capture_receipt.json", "kernel-bindings.json", "model.mlir")
    ]
    original = {path: path.read_bytes() for path in paths}
    try:
        trace = json.loads((package / "kernel-bindings.json").read_text())
        if damage == "profile":
            trace["execution_view_binding_profile"] = (
                "t2m.execution-checked-call-views.v1"
            )
        elif damage == "policy":
            trace["calls"][0]["tensor_views"] = "dense-row-major-v1"
        else:
            module = parse_shared((package / "model.mlir").read_text())
            if damage == "overflow-guard":
                from xdsl.dialects.builtin import StringAttr

                old = next(
                    op
                    for op in module.walk()
                    if op.name == "cf.assert"
                    and op.attributes.get("prov.op")
                    == StringAttr("runtime-view-contract")
                    and "overflow" in op.msg.data
                )
                old.parent.erase_op(old)
            else:
                old = next(
                    op
                    for op in module.walk()
                    if op.name == "tensor.expand_shape" and len(op.operands) == 2
                )
                one = next(
                    op.results[0]
                    for op in module.walk()
                    if op.name == "arith.constant"
                    and str(op.results[0].type) == "index"
                    and op.value.value.data == 1
                )
                old.operands = (old.operands[0], one)
            module.verify()
            (package / "model.mlir").write_text(serialize(module))
            trace["mlir_sha256"] = _record(package / "model.mlir")["sha256"]
        (package / "kernel-bindings.json").write_text(json.dumps(trace))
        rehash(package)
        with pytest.raises(ValueError) as producer_refusal:
            verify_bundle(package)
        output = tmp_path / "refused"
        with pytest.raises(ValueError) as consumer_refusal:
            check_merlin_interface(
                package,
                output,
                merlin_root=os.environ["T2M_MERLIN_ROOT"],
                python=os.environ["T2M_MERLIN_PYTHON"],
                result_mode="execution-owned",
            )
        assert not (output / "compiler").exists()
        (tmp_path / "refusal.json").write_text(
            json.dumps(
                {
                    "damage": damage,
                    "producer": str(producer_refusal.value),
                    "consumer": str(consumer_refusal.value),
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        for path, raw in original.items():
            path.write_bytes(raw)
    verify_bundle(package)
