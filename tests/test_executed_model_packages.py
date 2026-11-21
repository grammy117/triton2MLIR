"""Real internally sized models through package creation and Merlin staging."""

import json
import os
import shutil
import signal

import numpy as np
import pytest
import test_kernel_size_binding as kernel_models

from t2m.bundle import _record, verify_bundle
from t2m.consumer import check_merlin_interface, invoke_merlin_host, observe_merlin_host
from t2m.executed_model_binding import serialize
from t2m.model_bundle import write_model_bundle
from t2m.shared_tensor import parse_shared

parent = kernel_models.parent
pytestmark = pytest.mark.skipif(
    not kernel_models.REQUIRED,
    reason="existing compiler/model2MLIR/Merlin tools required",
)


def test_public_model_compiler_keeps_internally_sized_complete_results(selected, tmp_path):
    from public_model_helpers import compile_public, invoke_public

    _, _, package = selected
    product, _, bindings = compile_public(package, tmp_path, result_mode="execution-owned")
    assert len(bindings["initializers"]) == 1
    weight = np.asarray(-3, np.int32)
    for index, x in enumerate((np.arange(2, dtype=np.int32), np.arange(31, dtype=np.int32))):
        original = x + weight
        count = np.asarray([np.count_nonzero(original > 0)], np.int64)
        first = original[:count[0]].copy()
        expected = (first, first, original, x, count, weight, first[::2])
        invoke_public(product, [x], expected, tmp_path / f"run-{index}")


@pytest.fixture(scope="module", params=["tensor", "buffer", "guarded-buffer"])
def selected(tmp_path_factory, parent, request):
    root = tmp_path_factory.mktemp("executed-package-" + request.param)
    bindings = kernel_models.implementations(root / "translation", request.param)
    package = write_model_bundle(
        parent,
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
    return root, parent, package


def test_complete_original_model_qualifies_through_source_only_package_staging(
    selected,
):
    _root, parent, package = selected
    receipt = verify_bundle(package)
    assert receipt["schema"] == "t2m.bound-model-receipt.v2"
    assert receipt["qualification"] == "qualified"
    for name, destination in receipt["parent_artifact_bindings"].items():
        assert (package / destination).read_bytes() == (parent / name).read_bytes()
    signature = json.loads((package / "consumer/compiler/signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v6"
    assert "shape_program" not in signature
    correspondence = json.loads(
        (package / "consumer/compiler/source-correspondence.json").read_text()
    )
    assert correspondence["execution_binding"]["calls"] == 3
    assert correspondence["schema"] == "merlin.derived-source-correspondence.v2"
    assert len(signature["outputs"]) == 7
    with (
        np.load(package / "consumer/host/outputs.npz", allow_pickle=False) as outputs,
        np.load(parent / "goldens.npz", allow_pickle=False) as references,
    ):
        for i in range(7):
            out, expected = outputs[f"out{i}"], references[f"out{i}"]
            assert (out.dtype, out.shape, out.tobytes()) == (
                expected.dtype,
                expected.shape,
                expected.tobytes(),
            )


def test_published_package_reopens_and_runs_changed_content_without_compilation(
    selected, tmp_path
):
    _root, _parent, package = selected
    library = _record(package / "consumer/host/model.so")
    source = _record(package / "consumer/compiler/program.mlir")
    for capacity, weight in ((2, -100), (3, -1), (31, 0), (13, 100), (31, -29)):
        x = np.arange(capacity, dtype=np.int32)
        parameter = np.asarray(weight, np.int32)
        original = x + parameter
        count = np.asarray([np.count_nonzero(original > 0)], np.int64)
        first = original[: count[0]].copy()
        expected = (first, first, original, x, count, parameter, first[::2])
        arguments = tmp_path / f"arguments-{capacity}-{weight}.npz"
        np.savez(arguments, in0=parameter, in1=x)
        actual, report = invoke_merlin_host(
            package / "consumer",
            arguments,
            tmp_path / f"invoke-{capacity}-{weight}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert report["functional_inputs_preserved"]
        assert report["invocations"] == 2
        assert len(actual) == 7
        assert all(
            row["source_calls"] == 1
            and row["shapes"] == [list(value.shape) for value in expected]
            for row in report["result_observations"]
        )
        for out, value in zip(actual, expected, strict=True):
            assert (out.dtype, out.shape, out.tobytes()) == (
                value.dtype,
                value.shape,
                value.tobytes(),
            )
        assert _record(package / "consumer/host/model.so") == library
        assert _record(package / "consumer/compiler/program.mlir") == source


def test_original_source_domain_guards_abort_the_published_library(selected, tmp_path):
    _root, _parent, package = selected
    for capacity in (1, 32):
        arguments = tmp_path / f"invalid-{capacity}.npz"
        np.savez(
            arguments,
            in0=np.asarray(-3, np.int32),
            in1=np.arange(capacity, dtype=np.int32),
        )
        destination = tmp_path / f"failed-{capacity}"
        outcome = observe_merlin_host(
            package / "consumer",
            arguments,
            destination,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert outcome["returncode"] == -signal.SIGABRT
        assert not (destination / "outputs.npz").exists()


def test_compiler_staging_does_not_need_any_evaluator_or_existing_host_files(
    selected, tmp_path
):
    _root, _parent, package = selected
    copy = tmp_path / "source-only"
    shutil.copytree(package, copy)
    shutil.rmtree(copy / "consumer")
    for name in (
        "inputs.npz",
        "extra.npz",
        "golden.npy",
        "goldens.npz",
        "execution.json",
        "meta.json",
        "parent/meta.json",
    ):
        (copy / name).unlink(missing_ok=True)
    with pytest.raises(ValueError):
        verify_bundle(copy)
    interface = check_merlin_interface(
        copy,
        tmp_path / "source-only-stage",
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        result_mode="execution-owned",
    )
    assert interface["compiler_input_staging"] == "passed"
    members = interface["compiler_input_manifest"]["members"]
    assert not any(
        name.rsplit("/", 1)[-1]
        in {
            "inputs.npz",
            "golden.npy",
            "goldens.npz",
            "extra.npz",
            "execution.json",
            "meta.json",
        }
        for name in members
    )


def rehash(root):
    receipt = json.loads((root / "capture_receipt.json").read_text())
    for name in receipt["artifacts"]:
        receipt["artifacts"][name] = _record(root / name)
    (root / "capture_receipt.json").write_text(json.dumps(receipt))


@pytest.mark.parametrize("damage", ["body", "guard", "order", "located", "profile"])
def test_rehashed_changes_refuse_both_producer_replay_and_independent_staging(
    selected, tmp_path, damage
):
    _root, _parent, package = selected
    copy = tmp_path / "modified"
    shutil.copytree(package, copy)
    trace = json.loads((copy / "kernel-bindings.json").read_text())
    if damage in ("body", "guard"):
        from xdsl.dialects import arith, builtin

        module = parse_shared((copy / "model.mlir").read_text())
        if damage == "body":
            old = next(
                op
                for op in module.walk()
                if op.name == "arith.addi"
                and "prov.kernel_callsite" not in op.attributes
            )
            changed = arith.SubiOp(*old.operands)
            changed.attributes.update(old.attributes)
            changed.location = old.location
            old.parent.insert_op_before(changed, old)
            old.results[0].replace_all_uses_with(changed.result)
            old.parent.erase_op(old)
        else:
            guard = next(
                op
                for op in module.walk()
                if op.name == "cf.assert"
                and op.attributes.get("prov.family") == builtin.StringAttr("binding")
            )
            guard.parent.erase_op(guard)
        module.verify()
        (copy / "model.mlir").write_text(serialize(module))
        trace["mlir_sha256"] = _record(copy / "model.mlir")["sha256"]
    elif damage == "order":
        trace["calls"][1], trace["calls"][2] = trace["calls"][2], trace["calls"][1]
    elif damage == "located":
        trace["located_implementations"] = {}
    else:
        trace["schema"] = "t2m.model-kernel-binding.v1"
    (copy / "kernel-bindings.json").write_text(json.dumps(trace))
    rehash(copy)
    with pytest.raises(ValueError):
        verify_bundle(copy)
    destination = tmp_path / "rejected-stage"
    with pytest.raises(ValueError):
        check_merlin_interface(
            copy,
            destination,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
            result_mode="execution-owned",
        )
    assert not (destination / "compiler").exists()


def test_default_reader_requires_explicit_execution_selection(selected, tmp_path):
    _root, _parent, package = selected
    with pytest.raises(ValueError, match="explicit binding and result selection"):
        check_merlin_interface(
            package,
            tmp_path / "default-reader",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )


def test_coverage_records_the_actual_shared_host_qualification_backend(selected):
    from t2m.package_coverage import package_coverage

    _root, _parent, package = selected
    report = package_coverage([package])
    row = report["packages"][0]
    assert row["schema"] == "t2m.bound-model-receipt.v2"
    assert row["native_backend"] == "Merlin-host-execution-owned"
    assert row["independent_native_compiler_oracle"] is False
    assert row["merlin_host"]


def test_rehashed_result_observations_still_need_all_original_reference_words(
    selected, tmp_path
):
    _root, _parent, package = selected
    copy = tmp_path / "false-observation"
    shutil.copytree(package, copy)
    outputs_path = copy / "consumer/host/outputs.npz"
    with np.load(outputs_path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays["out0"].flat[0] += np.int32(1)
    np.savez(outputs_path, **arrays)
    evaluation = json.loads((copy / "consumer/host/evaluation.json").read_text())
    evaluation["outputs"] = _record(outputs_path)
    (copy / "consumer/host/evaluation.json").write_text(json.dumps(evaluation))
    host = json.loads((copy / "consumer/host/execution.json").read_text())
    host["evaluation"] = evaluation
    (copy / "consumer/host/execution.json").write_text(json.dumps(host))
    execution = json.loads((copy / "execution.json").read_text())
    execution["outputs"] = _record(outputs_path)
    (copy / "execution.json").write_text(json.dumps(execution))
    rehash(copy)
    with pytest.raises((AssertionError, ValueError)):
        verify_bundle(copy)


def test_failed_original_reference_comparison_never_publishes_a_qualified_package(
    parent, tmp_path
):
    capture = tmp_path / "contradicted-reference"
    shutil.copytree(parent, capture)
    with np.load(capture / "goldens.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays["out0"].flat[0] += np.int32(1)
    np.savez(capture / "goldens.npz", **arrays)
    np.save(capture / "golden.npy", arrays["out0"])
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    for name in ("golden.npy", "goldens.npz"):
        receipt["artifacts"][name] = _record(capture / name)
    (capture / "capture_receipt.json").write_text(json.dumps(receipt))
    bindings = kernel_models.implementations(tmp_path / "translation", "tensor")
    destination, work = tmp_path / "never-published", tmp_path / "qualification"
    with pytest.raises((AssertionError, ValueError)):
        write_model_bundle(
            capture,
            bindings,
            destination,
            work=work,
            python=os.environ["T2M_NATIVE_PYTHON"],
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            merlin_python=os.environ["T2M_MERLIN_PYTHON"],
            merlin_clang=os.environ["T2M_MERLIN_CLANG"],
            merlin_host=True,
            result_mode="execution-owned",
        )
    assert not destination.exists()
    candidate = work / "candidate"
    assert (candidate / "consumer/host/outputs.npz").is_file()
    assert (
        json.loads((candidate / "capture_receipt.json").read_text())["qualification"]
        == "source-only-unqualified"
    )
