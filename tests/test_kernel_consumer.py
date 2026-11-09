"""Actual Merlin standalone compilation, multiple outputs and refusal evidence."""

import json
import os
import shutil

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.cli import main, shared_demo
from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN, PYTHON, CLANG = (os.environ.get(name) for name in
                       ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not all((OPT, NATIVE, MERLIN, PYTHON, CLANG)),
                               reason="existing native and Merlin compiler environments required")


@pytest.fixture(scope="module", params=("tuple-scan", "descriptor-matmul"))
def captured(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("standalone-" + request.param)
    source = root / "capture"
    shared_demo(source, kernel=request.param, optimizer=OPT, python=NATIVE)
    return source


def qualify(source, out):
    return qualify_kernel(source, out, merlin_root=MERLIN, python=PYTHON,
                          compiler_python=NATIVE, clang=CLANG, rtol=1e-5, atol=6e-6)


def test_real_standalone_kernel_runs_through_merlin_with_source_lineage(captured, tmp_path, capsys):
    original = _record(captured / "capture_receipt.json")
    out = tmp_path / "qualified"
    qualify(captured, out)
    receipt = verify_kernel_qualification(out)
    assert _record(captured / "capture_receipt.json") == original
    verify_bundle(captured)
    stage = json.loads((out / "compiler/compile-inputs.json").read_text())
    assert stage["source_profile"] == "standalone-tensor-kernel"
    assert "inputs.npz" not in stage["members"] and "extra.npz" not in stage["members"]
    correspondence = json.loads((out / "compiler/source-correspondence.json").read_text())
    assert correspondence["source_entry"] == json.loads((captured / "spec.json").read_text())["entry"]
    assert correspondence["staged_entry"] == "forward"
    execution = json.loads((out / "host/execution.json").read_text())
    assert execution["evaluation"]["invocations"] == 2
    assert execution["evaluation"]["functional_inputs_preserved"] is True
    assert execution["evaluator_data_supplied_to_compiler"] is False
    assert "ownership-based-buffer-deallocation" in execution["pipeline"]
    assert execution["accelerator_execution"] == "not-performed"
    with np.load(out / "host/outputs.npz", allow_pickle=False) as actual:
        assert len(actual.files) == json.loads((captured / "capture_receipt.json").read_text())["materialized_abi"]["outputs"]
        np.testing.assert_allclose(actual["out0"], np.load(captured / "golden.npy"), rtol=1e-5, atol=6e-6)
        with np.load(captured / "extra.npz", allow_pickle=False) as extra:
            for key in extra.files:
                np.testing.assert_allclose(actual[key], extra[key], rtol=1e-5, atol=6e-6)
    assert main(["verify-merlin", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["artifacts_verified"] == len(receipt["artifacts"])
    from t2m.invocation import invoke_kernel, verify_invocation
    with np.load(captured / "extra.npz", allow_pickle=False) as extra:
        reference = {"out0": np.load(captured / "golden.npy", allow_pickle=False), **{k: extra[k].copy() for k in extra.files}}
    np.savez(tmp_path / "reference.npz", **reference)
    replay = tmp_path / "replay"
    invoke_kernel(out, captured / "inputs.npz", tmp_path / "reference.npz", replay,
                  merlin_root=MERLIN, python=PYTHON, rtol=1e-5, atol=6e-6)
    verify_invocation(replay)
    # Rehashing a changed report still cannot redirect its compiled-source relation.
    execution["source_program"]["sha256"] = "0" * 64
    (out / "host/execution.json").write_text(json.dumps(execution))
    receipt["artifacts"]["host/execution.json"] = _record(out / "host/execution.json")
    (out / "qualification.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="inconsistent source/result evidence"):
        verify_kernel_qualification(out)


def test_wrong_independent_reference_keeps_diagnostics_without_success_receipt(captured, tmp_path):
    source = tmp_path / "wrong-reference"
    shutil.copytree(captured, source)
    golden = np.load(source / "golden.npy", allow_pickle=False)
    np.save(source / "golden.npy", golden + 50, allow_pickle=False)
    receipt = json.loads((source / "capture_receipt.json").read_text())
    receipt["artifacts"]["golden.npy"] = _record(source / "golden.npy")
    (source / "capture_receipt.json").write_text(json.dumps(receipt))
    out = tmp_path / "failed"
    with pytest.raises(AssertionError):
        qualify(source, out)
    assert (out / "host/model.so").is_file()
    assert not (out / "qualification.json").exists()
