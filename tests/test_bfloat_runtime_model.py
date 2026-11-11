"""Real BF16 model inputs keep their original words through target execution."""

import json
import os

import numpy as np
import pytest

from test_bfloat_model import REQUIRED, NATIVE, MERLIN, CLANG, capture, translate
from test_spike_model import Q, TOOLS


@REQUIRED
@pytest.mark.skipif(not TOOLS, reason="existing selected target tools required")
@pytest.mark.parametrize("mode,legacy", [("observe", False), ("observe", True), ("matmul", False)])
def test_real_bfloat_inputs_use_source_words_outside_compiler_staging(tmp_path, mode, legacy):
    from t2m.bundle import _record, verify_bundle
    from t2m.model_bundle import write_model_bundle
    from t2m.model_runtime_inputs import ARCHIVE, RECORD

    parent = capture(tmp_path, mode, legacy=legacy)
    if mode == "observe" and not legacy:
        # A documented archive projection, not a different source execution:
        # legacy NaN values need not preserve the exact BF16 source payload.
        with np.load(parent / "inputs.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays["in0"][np.isnan(arrays["in0"])] = np.float32(np.nan)
        np.savez(parent / "inputs.npz", **arrays)
        receipt = json.loads((parent / "capture_receipt.json").read_text())
        receipt["artifacts"]["inputs.npz"] = _record(parent / "inputs.npz")
        receipt["archive_projection"] = "canonical legacy NaN input values; original source words/model/weights/references retained"
        (parent / "capture_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    binding = translate(tmp_path, mode, "guarded-buffer")
    model = tmp_path / "model"
    python = os.environ["T2M_MERLIN_PYTHON"]
    write_model_bundle(parent, [binding], model, work=tmp_path / "host", python=NATIVE,
        merlin_root=MERLIN, merlin_python=python, merlin_host=True, merlin_clang=CLANG)
    original = (model / "capture_receipt.json").read_bytes()
    target = tmp_path / "target"
    comparison = "values" if mode == "observe" and not legacy else "words"
    report = Q.qualify(model, target, merlin_root=MERLIN, python=python, comparison=comparison)
    assert report["evaluation"]["result_count"] == (6 if mode == "observe" else 1)
    assert Q.verify(target, merlin_root=MERLIN, python=python) == report
    assert (model / "capture_receipt.json").read_bytes() == original
    verify_bundle(model)
    projection = json.loads((target / RECORD).read_text())
    assert projection["arguments"][0]["word_witness"]["artifact"] == ("inputs.npz" if legacy else "tensor_words.npz")
    build = json.loads((target / "build.json").read_text())
    assert build["runtime_inputs"] == Q.pin(target / ARCHIVE)
    argv = json.loads((target / "build-process.json").read_text())["argv"]
    assert str(target / ARCHIVE) in argv
    assert all("golden" not in arg and "tensor_words" not in arg for arg in argv)
    assert not any((target / "compiler" / name).exists() for name in (ARCHIVE, RECORD, "tensor_words.npz", "inputs.npz"))
    expected, _ = Q.references(model)
    with np.load(target / "outputs.npz", allow_pickle=False) as actual:
        for i in ((0, 1, 3, 4) if mode == "observe" else (0,)):
            assert actual[f"out{i}"].tobytes() == expected[i].tobytes()

    # Rebinding the enclosing checksums cannot authorize different runtime words.
    original_archive = (target / ARCHIVE).read_bytes()
    original_build = (target / "build.json").read_bytes()
    original_report = (target / "qualification.json").read_bytes()
    with np.load(target / ARCHIVE, allow_pickle=False) as saved:
        changed = {key: saved[key].copy() for key in saved.files}
    changed["in0"].view(np.uint32).flat[0] ^= np.uint32(0x10000)
    try:
        np.savez(target / ARCHIVE, **changed)
        build["runtime_inputs"] = Q.pin(target / ARCHIVE)
        Q.write_json(target / "build.json", build)
        rebound = json.loads(original_report)
        for name in (ARCHIVE, "build.json"):
            rebound["artifacts"][name] = Q.pin(target / name)
        Q.write_json(target / "qualification.json", rebound)
        with pytest.raises(ValueError, match="original source words"):
            Q.verify(target, merlin_root=MERLIN, python=python)
    finally:
        (target / ARCHIVE).write_bytes(original_archive)
        (target / "build.json").write_bytes(original_build)
        (target / "qualification.json").write_bytes(original_report)
    assert Q.verify(target, merlin_root=MERLIN, python=python) == report
