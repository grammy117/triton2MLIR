"""Compiler-only custody and optional complete-model ISA execution evidence."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spike_model_qualification", ROOT / "scripts/qualify_spike_model.py")
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def stage(directory):
    directory.mkdir()
    (directory / "program.mlir").write_text("module {}")
    Q.write_json(directory / "compile-inputs.json", {"schema": "merlin.compile-inputs.v1",
        "members": {"program.mlir": Q.pin(directory / "program.mlir")["sha256"]},
        "excluded_evaluator_members": ["inputs.npz", "golden.npy"]})


@pytest.mark.parametrize("damage", ("payload", "extra", "alias", "symlink", "traversal", "evaluator"))
def test_compiler_projection_rejects_changed_or_unaccounted_members(tmp_path, damage):
    directory = tmp_path / "compiler"
    stage(directory)
    manifest = json.loads((directory / "compile-inputs.json").read_text())
    if damage == "payload":
        (directory / "program.mlir").write_text("changed")
    elif damage == "extra":
        (directory / "inputs.npz").write_bytes(b"runtime")
    elif damage == "alias":
        (directory / "model.mlir").write_text("different")
    elif damage == "symlink":
        saved = tmp_path / "saved.mlir"
        (directory / "program.mlir").rename(saved)
        (directory / "program.mlir").symlink_to(saved)
    elif damage == "traversal":
        manifest["members"]["../program.mlir"] = "0" * 64
    else:
        (directory / "golden.npy").write_bytes(b"reference")
        manifest["members"]["golden.npy"] = Q.pin(directory / "golden.npy")["sha256"]
    Q.write_json(directory / "compile-inputs.json", manifest)
    with pytest.raises(ValueError):
        Q.checked_stage(directory, model_alias=damage == "alias")


def _rebind_wire(directory, raw):
    (directory / "stdout.bin").write_bytes(raw)
    run = json.loads((directory / "run.json").read_text())
    run["stdout"] = Q.pin(directory / "stdout.bin")
    Q.write_json(directory / "run.json", run)
    report = json.loads((directory / "qualification.json").read_text())
    for name in ("stdout.bin", "run.json"):
        report["artifacts"][name] = Q.pin(directory / name)
    Q.write_json(directory / "qualification.json", report)


@pytest.mark.parametrize("rtol,atol", [(float("nan"), 0), (0, float("inf")), (-1, 0)])
def test_invalid_policy_cannot_qualify_even_empty_outputs(tmp_path, rtol, atol):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        Q.compare(tmp_path / "source", tmp_path / "result", rtol, atol)


def _change_frame(raw, name):
    start = raw.index(f"OUT_BIN_BEGIN v1 {name} ".encode())
    payload_start = raw.index(b"\n", start) + 1
    count = int(raw[start:payload_start].split()[-1])
    assert count > 0
    payload = bytearray(raw[payload_start:payload_start + count])
    payload[0] ^= 1
    checksum = 0xCBF29CE484222325
    for byte in payload:
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    footer_end = raw.index(b"\n", payload_start + count) + 1
    return raw[:payload_start] + payload + f"OUT_BIN_END v1 {checksum:016x}\n".encode() + raw[footer_end:]


CAPTURES = [Path(p) for p in os.environ.get("T2M_SPIKE_MODEL_CAPTURES", "").split(os.pathsep) if p]
TOOLS = all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON",
    "MERLIN_COMPILER_PYTHON", "MERLIN_CLANG", "MERLIN_RISCV_GCC", "MERLIN_SPIKE", "MERLIN_NM"))


@pytest.mark.skipif(not (CAPTURES and TOOLS), reason="explicit bound models and existing cross compiler/Spike required")
@pytest.mark.parametrize("capture", CAPTURES or [None], ids=[p.name for p in CAPTURES] or ["unselected"])
def test_complete_model_executes_and_retained_wire_is_independently_checked(tmp_path, capture):
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    target = tmp_path / "target"
    report = Q.qualify(capture, target, **options)
    assert report["status"] == "passed"
    expected, _ = Q.references(capture)
    with np.load(target / "outputs.npz", allow_pickle=False) as actual:
        assert set(actual.files) == {f"out{i}" for i in range(len(expected))}
        for i, value in enumerate(expected):
            observed = actual[f"out{i}"]
            assert (observed.shape, observed.dtype, observed.tobytes()) == (value.shape, value.dtype, value.tobytes())
    run = json.loads((target / "run.json").read_text())
    assert all(row["passed"] for row in run["readonly_inputs"]["arguments"])
    assert run["accelerator_execution"] == "not-performed"
    build = json.loads((target / "build.json").read_text())
    assert build["reference_values_supplied_to_build"] is False
    assert build["runtime_inputs_embedded_in_elf"] is True
    originals = {name: (target / name).read_bytes() for name in ("stdout.bin", "run.json", "qualification.json")}
    for name, message in (("out0", "differ from raw frames"), ("readonly_arg_0", "changed during forward")):
        try:
            _rebind_wire(target, _change_frame(originals["stdout.bin"], name))
            # Even a valid recomputed checksum and updated outer pins cannot
            # substitute different values for the actual retained output/input.
            with pytest.raises(ValueError, match=message):
                Q.verify(target, **options)
        finally:
            for filename, raw in originals.items():
                (target / filename).write_bytes(raw)
    Q.verify(target, **options)
    injected = target / "compiler/inputs.npz"
    try:
        injected.write_bytes(b"evaluator data")
        with pytest.raises(ValueError, match="undeclared members"):
            Q.verify(target, **options)
    finally:
        injected.unlink()
    Q.verify(target, **options)
