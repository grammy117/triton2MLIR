#!/usr/bin/env python3
"""Check the frozen release evidence; observations alone cannot close v1.

The semantic audit is an explicit source/profile review bound to actual files
and qualified cases. This gate checks its custody and obligations, rather than
claiming a formal proof of arbitrary source programs from passing tests.
"""

import argparse
import hashlib
import json
import os
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("evidence must be a direct regular file: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("bounded release JSON/XML exceeded 16MiB: " + str(path))
    digest(path)
    return json.loads(path.read_text())


def local(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("evidence member must stay within its declared root")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("evidence member escaped through a symlink")
    return result


def acceptance(run, manifest):
    if read(run / "manifest.json") != manifest:
        raise ValueError("test run changed the frozen acceptance manifest")
    if read(run / "process.json")["returncode"] != 0:
        raise ValueError("acceptance process failed")
    cases = manifest["cases"]
    command = read(run / "command.json")
    if (command[1:3] != ["-m", "pytest"] or command[-len(cases):] != cases
            or any(option in command[3:] for option in ("-k", "-m", "--deselect"))):
        raise ValueError("acceptance command did not request the complete exact case roster")
    path = run / "pytest.xml"
    digest(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("bounded JUnit exceeded 16MiB")
    rows = list(ET.parse(path).getroot().iter("testcase"))
    actual = [row.attrib["classname"].replace(".", "/") + ".py::" + row.attrib["name"] for row in rows]
    if len(actual) != len(cases) or set(actual) != set(cases):
        raise ValueError("JUnit omitted, repeated or substituted a frozen case")
    if any(row.find(tag) is not None for row in rows for tag in ("failure", "error", "skipped")):
        raise ValueError("a required acceptance case failed, errored or skipped")
    return set(actual)


def semantic_audit(path, policy, manifest, passed, roots):
    audit = read(path)
    definitions = runpy.run_path(str(ROOT / "t2m/ttir_definitions.py"))
    required = set(definitions["OPERATIONS"]) - {"tt.elementwise_inline_asm"}
    if (audit.get("schema") != "t2m.semantic-closure-audit.v1"
            or audit.get("policy_sha256") != digest(ROOT / "docs/v1_policy.json")
            or audit.get("acceptance_sha256") != digest(ROOT / "tests/data/v1_acceptance.json")
            or set(audit.get("operations", {})) != required):
        raise ValueError("semantic audit must bind the exact policy/manifest and all 49 generic operations")
    sources = audit.get("implementation_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("semantic review has no current implementation source custody")
    for source in sources:
        root = roots.get(source["repository"])
        if root is None or digest(local(root, source["path"])) != source["sha256"]:
            raise ValueError("semantic review implementation source changed")
    positive = {case for key, cases in manifest["workflows"].items()
                if key.startswith(("K", "M")) for case in cases} & passed
    for name, row in audit["operations"].items():
        if (row.get("route") not in ("portable", "portable-with-contract", "portable-fallback")
                or not isinstance(row.get("profile"), str) or not row["profile"].strip()
                or row.get("unresolved_generic_forms") != []
                or not row.get("qualified_cases") or not set(row["qualified_cases"]) <= positive):
            raise ValueError("generic operation lacks a closed executable profile and qualified cases: " + name)
        # A case-name observation or an author's 'passed' boolean is insufficient.
        # Retain the actual package's complete byte/source/ABI/reference checks.
        from t2m.bundle import verify_bundle

        packages = row.get("packages")
        if not isinstance(packages, list) or not packages:
            raise ValueError("generic semantic review lacks executable package evidence: " + name)
        observed = False
        for package in packages:
            directory = local(roots["triton2mlir"], package["path"])
            if digest(directory / "capture_receipt.json") != package["receipt_sha256"]:
                raise ValueError("semantic package receipt changed: " + name)
            receipt = verify_bundle(directory)
            if "qualification" in package:
                from t2m.kernel_consumer import verify_kernel_qualification

                qualification = local(roots["triton2mlir"], package["qualification"])
                qualified = verify_kernel_qualification(qualification)
                if qualified["source_receipt"]["sha256"] != package["receipt_sha256"]:
                    raise ValueError("Merlin qualification belongs to another source package: " + name)
            elif "consumer/host/execution.json" not in receipt["artifacts"]:
                raise ValueError("semantic package has no byte-bound Merlin host qualification: " + name)
            members = receipt["artifacts"]
            for member in members:
                if Path(member).name not in ("provenance.json", "allocation-inventory.json", "allocation-protection.json"):
                    continue
                inventory = read(local(directory, member))
                operations = inventory.get("source_operations", inventory.get("operations",
                    inventory.get("source_graph", {}).get("operations", [])))
                observed |= any(operation.get("opcode") == name for operation in operations)
        if not observed:
            raise ValueError("qualified package inventories do not contain the reviewed original operation: " + name)
    if audit.get("limitations_sha256") != digest(ROOT / "docs/limitations.md"):
        raise ValueError("semantic review did not bind the published limitations")


def check(run, *, audit=None, replay=None, merlin_root=None, model2mlir_root=None):
    run = Path(run).resolve()
    policy = read(ROOT / "docs/v1_policy.json")
    manifest = read(ROOT / "tests/data/v1_acceptance.json")
    definitions = runpy.run_path(str(ROOT / "t2m/ttir_definitions.py"))
    checks, passed = [], set()

    def gate(name, function):
        try:
            function()
        except (ValueError, KeyError, OSError, AssertionError, ET.ParseError) as error:
            checks.append({"gate": name, "status": "open", "reason": str(error)})
        else:
            checks.append({"gate": name, "status": "passed"})

    def pinned_policy():
        if (policy["definitions"] != definitions["SOURCE"]
                or set(policy["operations"]) != set(definitions["OPERATIONS"])
                or policy["generic_exclusion_allowed"] is not False):
            raise ValueError("release policy changed the pinned definitions or generic requirement")
        for name, row in policy["operations"].items():
            if row["requirement"] != ("optional-source-language-provider" if name == "tt.elementwise_inline_asm"
                                      else "mandatory-generic-computation"):
                raise ValueError("release policy dropped a generic obligation: " + name)
        if (not 0 < len(manifest["cases"]) == manifest["case_count"] <= manifest["maximum_cases"] <= 128
                or len(set(manifest["cases"])) != manifest["case_count"]):
            raise ValueError("acceptance roster must be unique, nonempty and bounded to 128 cases")
        for name, expected in manifest["fixture_sources"].items():
            if digest(local(ROOT, name)) != expected:
                raise ValueError("frozen acceptance fixture changed: " + name)
        if not all(manifest["workflows"].get("M" + str(i)) for i in range(1, 8)):
            raise ValueError("acceptance roster omitted a required model workflow")

    gate("pinned-policy-and-fixture-custody", pinned_policy)

    def complete_execution():
        passed.update(acceptance(run, manifest))

    gate("complete-acceptance-execution", complete_execution)
    roots = {"triton2mlir": ROOT, "merlin": Path(merlin_root).resolve() if merlin_root else None,
             "model2mlir": Path(model2mlir_root).resolve() if model2mlir_root else None}
    gate("generic-semantic-profile-audit", lambda: semantic_audit(
        Path(audit) if audit else run / "semantic-audit.json", policy, manifest, passed, roots))

    def baseline():
        if replay is None or roots["merlin"] is None:
            raise ValueError("the final Merlin regression replay and its source root are required")
        evidence = read(replay)
        if evidence.get("schema") != "merlin.ttir-layout-replay.v1" or evidence.get("status") != "passed" or evidence.get("returncode") != 0:
            raise ValueError("Merlin baseline replay did not pass")
        sources = evidence.get("consumer_source_sha256")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("Merlin baseline replay has no implementation source custody")
        for name, expected in sources.items():
            # The source snapshot includes intentional repository resource
            # symlinks. Bind their actual in-root contents; package members
            # above retain their stricter direct-file contract.
            if digest(local(roots["merlin"], name).resolve(strict=True)) != expected:
                raise ValueError("Merlin implementation differs from the baseline replay: " + name)

    gate("baseline-replay-source-custody", baseline)
    return {"schema": "t2m.closure-check.v1", "closure_status": "closed" if all(row["status"] == "passed" for row in checks) else "open",
            "checks": checks, "required_cases": manifest["case_count"], "passed_cases": len(passed),
            "interpretation": "Pinned release/profile evidence; no universal source-equivalence or accelerator-performance proof."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path)
    parser.add_argument("--merlin-replay", type=Path)
    parser.add_argument("--merlin-root", default=os.environ.get("T2M_MERLIN_ROOT"))
    parser.add_argument("--model2mlir-root", default=os.environ.get("T2M_MODEL2MLIR_ROOT"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = check(arguments.run, audit=arguments.semantic_audit, replay=arguments.merlin_replay,
                   merlin_root=arguments.merlin_root, model2mlir_root=arguments.model2mlir_root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if report["closure_status"] == "closed" else 1)
