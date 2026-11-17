"""Real v6 TTIR packages reach the ordinary model builder and actual Spike."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_compact_model import REQUIRED, parent  # noqa: F401
from test_compact_model import selected as captured_selection
from test_spike_model import Q, _change_frame

from t2m.bundle import verify_bundle

TOOLS = REQUIRED and all(os.environ.get(name) for name in ("MERLIN_RISCV_GCC", "MERLIN_SPIKE", "MERLIN_NM"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing TTIR/model/Merlin/cross/Spike tools required")
selected = captured_selection


@pytest.fixture(scope="module")
def target(selected, tmp_path_factory):
    _root, _parent, _pack, package = selected
    root = tmp_path_factory.mktemp("compact-target")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        report = Q.qualify(package, root / "target", result_mode="execution-owned", comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert report["schema"] == "t2m.spike-owned-result-qualification.v1"
    return root / "target", package


def test_actual_model_package_preserves_all_original_results_on_target(target):
    target, package = target
    receipt = verify_bundle(package)
    assert receipt["schema"] == "t2m.bound-model-receipt.v2"
    report = Q.verify(target, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert report["evaluation"]["result_count"] == 7
    run = json.loads((target / "run.json").read_text())
    assert run["readback"]["functional_inputs_preserved"]
    assert run["readback"]["profile"] == "merlin.execution-owned-readback.v1"
    assert run["accelerator_execution"] == "not-performed"
    build = json.loads((target / "build.json").read_text())
    assert build["reference_values_supplied_to_build"] is False
    assert build["compiler_directory_contains_evaluator_data"] is False
    assert build["build"]["allocation_context"] == "serial-nonconcurrent"
    compiler = json.loads((target / "compiler/compile-inputs.json").read_text())
    assert all("golden" not in name and not name.endswith(".npz") for name in compiler["members"])
    with np.load(target / "outputs.npz", allow_pickle=False) as outputs, \
            np.load(package / "goldens.npz", allow_pickle=False) as references:
        assert len(outputs.files) == len(references.files) == 7
        for i in range(7):
            actual, wanted = outputs[f"out{i}"], references[f"out{i}"]
            assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())


def invocation(root, words, weight, *, wrong=False):
    x, parameter = np.asarray(words, np.int32), np.asarray(weight, np.int32)
    original = np.add(x, parameter, dtype=np.int32)
    packed = original[original > 0]
    results = (packed, packed, original, x, np.asarray([packed.size], np.int64), parameter, packed[::2])
    np.savez(root / "inputs.npz", in0=parameter, in1=x)
    if wrong:
        results = (packed + 1, *results[1:])
    np.savez(root / "references.npz", **{f"out{i}": value for i, value in enumerate(results)})
    return root / "inputs.npz", root / "references.npz"


@pytest.mark.parametrize("words,weight", [([0, -1] * 6 + [0], 0), (list(range(31)), 100),
    ([0, -1, 1, -(1 << 31), (1 << 31) - 1] * 6 + [7], 1)])
def test_changed_target_inputs_keep_empty_full_and_overflowing_model_results(target, tmp_path, words, weight):
    _target, package = target
    inputs, reference = invocation(tmp_path, words, weight)
    report = Q.qualify(package, tmp_path / "target", result_mode="execution-owned", comparison="words",
        inputs=inputs, reference=reference, merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"])
    assert report["evaluation"]["result_count"] == 7


def test_real_source_domain_failure_cannot_publish_target_success(target, tmp_path):
    _target, package = target
    inputs, reference = invocation(tmp_path, list(range(32)), 0)
    with pytest.raises(ValueError, match="whole-model run failed"):
        Q.qualify(package, tmp_path / "target", result_mode="execution-owned", comparison="words",
            inputs=inputs, reference=reference, merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"])
    directory = tmp_path / "target"
    process = json.loads((directory / "target-process.json").read_text())
    assert process["returncode"] != 0
    raw = (directory / "stdout.bin").read_bytes()
    assert b"exported shape domain:" in raw and b"OUT_BIN_BEGIN" not in raw and b"DONE" not in raw
    assert not (directory / "qualification.json").exists()


def test_complete_wrong_reference_cannot_publish_target_success(target, tmp_path):
    _target, package = target
    inputs, reference = invocation(tmp_path, [1, 6, -2], 0, wrong=True)
    with pytest.raises(ValueError, match="complete reference words"):
        Q.qualify(package, tmp_path / "target", result_mode="execution-owned", comparison="words",
            inputs=inputs, reference=reference, merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"])
    assert not (tmp_path / "target/qualification.json").exists()


@pytest.mark.parametrize("damage", ["payload", "correspondence", "extra"])
def test_rehashed_target_records_cannot_replace_observations_or_source_reconstruction(target, damage):
    directory, _package = target
    original = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    try:
        report = json.loads((directory / "qualification.json").read_text())
        if damage == "payload":
            raw = _change_frame((directory / "stdout.bin").read_bytes(), "out0")
            (directory / "stdout.bin").write_bytes(raw)
            process = json.loads((directory / "target-process.json").read_text())
            process["stdout"] = Q.pin(directory / "stdout.bin")
            Q.write_json(directory / "target-process.json", process)
            run = json.loads((directory / "run.json").read_text())
            run["stdout"] = Q.pin(directory / "stdout.bin")
            run["target_process"] = Q.pin(directory / "target-process.json")
            Q.write_json(directory / "run.json", run)
        elif damage == "correspondence":
            for parent in ("compiler", "image-input"):
                correspondence = directory / parent / "source-correspondence.json"
                value = json.loads(correspondence.read_text())
                value["scope"] = "a rehashed declaration cannot replace independent source reconstruction"
                Q.write_json(correspondence, value)
                manifest = json.loads((directory / parent / "compile-inputs.json").read_text())
                manifest["members"]["source-correspondence.json"] = Q.pin(correspondence)["sha256"]
                Q.write_json(directory / parent / "compile-inputs.json", manifest)
        else:
            (directory / "unaccounted.txt").write_text("not part of the original target evidence\n")
        # Deliberately repair outer byte custody. Semantic/source reconstruction
        # and retained-wire checks must still reject these changed claims.
        report["artifacts"] = {name: Q.pin(directory / name) for name in report["artifacts"]}
        Q.write_json(directory / "qualification.json", report)
        with pytest.raises(ValueError, match="owned-result"):
            Q.verify(directory, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    finally:
        for path in directory.rglob("*"):
            if path.is_file() and path not in original:
                path.unlink()
        for path, raw in original.items():
            path.write_bytes(raw)
    Q.verify(directory, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
