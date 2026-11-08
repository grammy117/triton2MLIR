"""External stage transport contracts; real compiler checks are explicitly opt-in."""

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from t2m.diagnostics import ConversionError
from t2m.upstream import lower_with_shared


def process_fixture(tmp_path, body):
    """A process protocol fixture, never a compiler/correctness oracle."""
    tool = tmp_path / "process_fixture"
    tool.write_text(f"#!{sys.executable}\n" + body + "\n")
    tool.chmod(0o700)
    return tool


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "empty.ttir"
    path.write_text("module {}\n")
    return path


def test_unavailable_upstream_does_not_publish_any_artifact(source, tmp_path):
    out = tmp_path / "output"
    with pytest.raises(ConversionError, match="no source build is started automatically"):
        lower_with_shared(source, out, optimizer=tmp_path / "not-installed")
    assert not out.exists()


@pytest.mark.parametrize("stage", ["unknown", "--triton-to-structured=use-unsafe-mask=true"])
def test_arbitrary_or_unsafe_stage_selection_is_rejected(source, tmp_path, stage):
    with pytest.raises(ConversionError, match="unknown upstream stage"):
        lower_with_shared(source, tmp_path / "out", stage=stage)


def test_failed_optimizer_cannot_become_a_complete_stage(source, tmp_path):
    tool = process_fixture(tmp_path, "import sys; sys.stderr.write('incompatible IR version'); sys.exit(7)")
    out = tmp_path / "output"
    with pytest.raises(ConversionError, match="incompatible IR version"):
        lower_with_shared(source, out, optimizer=tool)
    assert not (out / "stage.json").exists()
    assert not json.loads((out / "failure.json").read_text())["portable_conversion_complete"]


def test_process_transport_records_bytes_but_never_claims_tensor_support(source, tmp_path):
    tool = process_fixture(tmp_path, "print('module {}')")
    out = tmp_path / "output"
    pin = hashlib.sha256(tool.read_bytes()).hexdigest()
    result = lower_with_shared(source, out, optimizer=tool, expected_optimizer_sha256=pin)
    assert result["stage_completed"]
    assert result["optimizer"]["binary_hash_pin_checked"]
    assert not result["optimizer"]["source_revision_verified"]
    assert not result["portable_conversion_complete"] and not result["tensor_abi_compatible"]
    assert result["source_correspondence"] == "not-established"
    assert result["numeric_equivalence"] == "not-qualified"
    assert result["limits"]["mlir_threading"] == "disabled"
    assert result["artifacts"]["lowered.mlir"]["sha256"] == hashlib.sha256((out / "lowered.mlir").read_bytes()).hexdigest()
    assert not (out / "capture_receipt.json").exists() and not (out / "model.mlir").exists()
    with pytest.raises(FileExistsError):
        lower_with_shared(source, out, optimizer=tool)


def test_binary_pin_mismatch_precedes_invocation(source, tmp_path):
    tool = process_fixture(tmp_path, "raise RuntimeError('must not run')")
    with pytest.raises(ConversionError, match="hash does not match"):
        lower_with_shared(source, tmp_path / "out", optimizer=tool, expected_optimizer_sha256="0" * 64)
    assert not (tmp_path / "out").exists()


def test_timeout_and_empty_output_fail_closed(source, tmp_path):
    sleeping = process_fixture(tmp_path, "import time; time.sleep(2)")
    with pytest.raises(ConversionError, match="exceeded"):
        lower_with_shared(source, tmp_path / "timed", optimizer=sleeping, timeout=0.05)
    empty = process_fixture(tmp_path, "pass")
    with pytest.raises(ConversionError, match="no output"):
        lower_with_shared(source, tmp_path / "empty", optimizer=empty)


def test_output_file_size_is_bounded(source, tmp_path):
    tool = process_fixture(tmp_path, "import sys; sys.stdout.write('x' * 8192); sys.stdout.flush()")
    out = tmp_path / "bounded"
    with pytest.raises(ConversionError, match="failed with exit code"):
        lower_with_shared(source, out, optimizer=tool, output_limit=1024)
    assert (out / "lowered.mlir").stat().st_size <= 1024


@pytest.mark.skipif(not os.environ.get("TRITON_SHARED_OPT"), reason="existing upstream optimizer not configured")
@pytest.mark.parametrize("stage,marker", [("structured", '"tts.make_tptr"'),
                                        ("tensor-ir", '"tt.func"'),
                                        ("tensor-compute", '"linalg.generic"'),
                                        ("full", "memref")])
def test_real_upstream_multidimensional_fixture(tmp_path, stage, marker):
    fixture = Path(__file__).parent / "fixtures/triton_shared/addptr_2d_example.mlir"
    result = lower_with_shared(fixture, tmp_path / stage, stage=stage)
    assert marker in (tmp_path / stage / "lowered.mlir").read_text()
    assert result["stage_completed"] and not result["portable_conversion_complete"]
