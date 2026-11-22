"""Incomplete execution records and opcode observations cannot close release."""

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_checker", ROOT / "scripts/check_v1_closure.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def ledger(root):
    """A JUnit reader fixture, not compiler/numeric qualification evidence."""
    manifest = json.loads((ROOT / "tests/data/v1_acceptance.json").read_text())
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "process.json").write_text(json.dumps({"returncode": 0}))
    (root / "command.json").write_text(json.dumps(["python", "-m", "pytest", *manifest["cases"]]))
    tree = ET.Element("testsuites")
    suite = ET.SubElement(tree, "testsuite")
    for case in manifest["cases"]:
        path, name = case.split("::", 1)
        ET.SubElement(suite, "testcase", classname=path.removesuffix(".py").replace("/", "."), name=name)
    ET.ElementTree(tree).write(root / "pytest.xml")
    return manifest


@pytest.mark.parametrize("damage", ("missing-case", "skipped-case", "failed-process", "changed-selection"))
def test_execution_reader_cannot_count_partial_skipped_failed_or_different_cases(tmp_path, damage):
    manifest = ledger(tmp_path)
    assert CHECKER.acceptance(tmp_path, manifest) == set(manifest["cases"])
    if damage == "failed-process":
        (tmp_path / "process.json").write_text(json.dumps({"returncode": 1}))
    elif damage == "changed-selection":
        (tmp_path / "command.json").write_text(json.dumps(["python", "-m", "pytest", "-k", "some", *manifest["cases"]]))
    else:
        tree = ET.parse(tmp_path / "pytest.xml")
        suite = tree.getroot().find("testsuite")
        if damage == "missing-case":
            suite.remove(suite.find("testcase"))
        else:
            ET.SubElement(suite.find("testcase"), "skipped")
        tree.write(tmp_path / "pytest.xml")
    with pytest.raises(ValueError):
        CHECKER.acceptance(tmp_path, manifest)


def test_even_a_complete_case_ledger_cannot_replace_semantic_audit_and_source_replay(tmp_path, monkeypatch):
    manifest = ledger(tmp_path)
    # Keep the reader's unresolved-gap obligation independent of which real
    # compiler capability is next to close. This is no execution evidence.
    read = CHECKER.read
    gap_path = ROOT / "tests/data/v1_generic_gaps.json"
    fixture = read(gap_path)
    fixture["gaps"] = [next(row for row in fixture["closed_gaps"] if row["id"] == "subbyte-atomic-storage")]
    monkeypatch.setattr(CHECKER, "read", lambda path: fixture if Path(path) == gap_path else read(path))
    (tmp_path / "semantic-audit.json").write_text(json.dumps({"closure_status": "closed"}))
    result = CHECKER.check(tmp_path)
    assert result["passed_cases"] == len(manifest["cases"])
    assert result["closure_status"] == "open"
    gates = {row["gate"]: row["status"] for row in result["checks"]}
    assert gates["complete-acceptance-execution"] == "passed"
    assert gates["generic-semantic-profile-audit"] == gates["baseline-replay-source-custody"] == "open"
    known = next(row for row in result["checks"] if row["gate"] == "confirmed-generic-source-gaps")
    assert known["status"] == "open" and "subbyte-atomic-storage" in known["reason"]


def test_success_label_without_replay_source_custody_does_not_pass_baseline(tmp_path):
    ledger(tmp_path)
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"schema": "merlin.ttir-layout-replay.v1", "status": "passed",
                                  "returncode": 0, "consumer_source_sha256": {}}))
    result = CHECKER.check(tmp_path, replay=replay, merlin_root=ROOT)
    baseline = next(row for row in result["checks"] if row["gate"] == "baseline-replay-source-custody")
    assert baseline["status"] == "open" and "no implementation source custody" in baseline["reason"]
