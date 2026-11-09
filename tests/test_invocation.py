"""New runtime values use the same qualified library without a compiler."""

import json
import os
import shutil

import numpy as np
import pytest

from t2m.bundle import _record
from t2m.cli import main, shared_demo
from t2m.invocation import invoke_kernel, verify_invocation
from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN, PYTHON, CLANG = (os.environ.get(name) for name in
    ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not all((OPT, NATIVE, MERLIN, PYTHON, CLANG)), reason="existing native/Merlin toolchains required")


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    root = tmp_path_factory.mktemp("runtime-reuse")
    capture, out = root / "capture", root / "qualified"
    shared_demo(capture, kernel="runtime-affine", optimizer=OPT, python=NATIVE)
    qualify_kernel(capture, out, merlin_root=MERLIN, python=PYTHON, compiler_python=NATIVE, clang=CLANG)
    return out


def arguments(root, alpha=-2, bias=17):
    x = np.arange(-6, 7, dtype=np.float32)
    inputs, reference = root / "inputs.npz", root / "reference.npz"
    np.savez(inputs, in0=x, in1=np.asarray(alpha, dtype=np.float32), in2=np.asarray(bias, dtype=np.float32))
    np.savez(reference, out0=x * np.float32(alpha) + np.float32(bias))
    return inputs, reference


@pytest.mark.parametrize("alpha,bias", [(0, -9), (-2, 17)])
def test_changed_scalars_reuse_exact_library_and_preserve_original_qualification(compiled, tmp_path, monkeypatch, capsys, alpha, bias):
    before = verify_kernel_qualification(compiled)
    library = _record(compiled / "host/model.so")
    inputs, reference = arguments(tmp_path, alpha, bias)
    # Executing a qualified object needs no native compiler/interpreter lookup.
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", str(tmp_path / "unavailable-compiler"))
    monkeypatch.setenv("MERLIN_CLANG", str(tmp_path / "unavailable-clang"))
    out = tmp_path / "invocation"
    report = invoke_kernel(compiled, inputs, reference, out, merlin_root=MERLIN, python=PYTHON)
    verify_invocation(out)
    assert report["native_compilation"] == "reused-qualified-library"
    assert report["evaluation"]["shared_library"] == library
    assert report["evaluation"]["arguments"] == _record(inputs)
    assert verify_kernel_qualification(compiled) == before
    assert _record(compiled / "host/model.so") == library
    assert not (out / "host/lowering").exists()
    assert main(["verify-merlin-invocation", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["artifacts_verified"] == 8
    with pytest.raises(FileExistsError):
        invoke_kernel(compiled, inputs, reference, out, merlin_root=MERLIN, python=PYTHON)
    # Rehashing a result report cannot change its program or library relation.
    evaluation = json.loads((out / "evaluation/evaluation.json").read_text())
    evaluation["shared_library"]["sha256"] = "0" * 64
    (out / "evaluation/evaluation.json").write_text(json.dumps(evaluation))
    report["evaluation"] = evaluation
    report["artifacts"]["evaluation/evaluation.json"] = _record(out / "evaluation/evaluation.json")
    (out / "invocation.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="qualified library/program"):
        verify_invocation(out)


@pytest.mark.parametrize("change", ["dtype", "shape", "keys", "reference", "reference_keys"])
def test_invalid_runtime_arguments_and_references_cannot_publish_success(compiled, tmp_path, change):
    inputs, reference = arguments(tmp_path)
    with np.load(inputs, allow_pickle=False) as data:
        values = {name: data[name].copy() for name in data.files}
    if change == "dtype":
        values["in1"] = np.asarray(-2, dtype=np.float64)
    elif change == "shape":
        values["in1"] = np.asarray([-2], dtype=np.float32)
    elif change == "keys":
        values.pop("in2")
    elif change == "reference":
        np.savez(reference, out0=np.full(13, 99, dtype=np.float32))
    else:
        np.savez(reference, out1=np.zeros(13, dtype=np.float32))
    np.savez(inputs, **values)
    out = tmp_path / "failed"
    with pytest.raises((ValueError, AssertionError)):
        invoke_kernel(compiled, inputs, reference, out, merlin_root=MERLIN, python=PYTHON)
    assert not (out / "invocation.json").exists()
    verify_kernel_qualification(compiled)


def test_changed_source_library_is_rejected_before_invocation(compiled, tmp_path):
    source = tmp_path / "changed-qualification"
    shutil.copytree(compiled, source)
    library = source / "host/model.so"
    library.write_bytes(library.read_bytes() + b"changed")
    inputs, reference = arguments(tmp_path)
    out = tmp_path / "invocation"
    with pytest.raises(ValueError, match="integrity mismatch"):
        invoke_kernel(source, inputs, reference, out, merlin_root=MERLIN, python=PYTHON)
    assert not out.exists()
