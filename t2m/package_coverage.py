"""Join verified package observations to a pinned opcode denominator.

This inspects retained evidence; it executes no compiler/kernel and grants no
new correctness authority. Source inventories remain producer declarations.
An observed opcode/type/attribute composition is never universal coverage.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .bundle import _record, verify_bundle
from .kernel_consumer import verify_kernel_qualification
from .ttir_definitions import OPERATIONS, SOURCE


def _json(path):
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("coverage JSON exceeds its 32 MiB budget")
    return json.loads(path.read_text())


def _member(root, name):
    relative = Path(name)
    if (relative.is_absolute() or ".." in relative.parts or not relative.parts
            or relative.as_posix() != name
            or any(root.joinpath(*relative.parts[:i]).is_symlink() for i in range(len(relative.parts) + 1))):
        raise ValueError("coverage requires local nonsymlink artifact paths")
    return root / relative


def _receipt(root, filename):
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("coverage requires local nonsymlink artifact paths")
    receipt = _json(_member(root, filename))
    if not isinstance(receipt, dict) or not isinstance(receipt.get("artifacts"), dict):
        raise ValueError("coverage requires an artifact receipt")
    for name in receipt.get("artifacts", {}):
        _member(root, name)
    return receipt


def operation_observations(cases):
    """Index source declarations; normalized descendants are a separate graph."""
    rows = {name: {"definition": definition, "cases": [], "merlin_cases": [], "forms": []}
            for name, definition in OPERATIONS.items()}
    unknown, other, forms = {}, {}, defaultdict(dict)
    for case in cases:
        operations = case["operations"]
        if not isinstance(operations, list) or len(operations) > 200_000:
            raise ValueError("coverage source inventory exceeds its operation budget")
        ids = set()
        for op in operations:
            if (not isinstance(op, dict) or not isinstance(op.get("id"), str) or op["id"] in ids
                    or not isinstance(op.get("opcode"), str)
                    or any(not isinstance(op.get(key), list) or any(not isinstance(t, str) for t in op[key])
                           for key in ("input_types", "output_types"))
                    or any(op.get(key) is not None and not isinstance(op[key], dict) for key in ("attributes", "properties"))):
                raise ValueError("coverage needs unique typed source operation declarations")
            ids.add(op["id"])
            name = op["opcode"]
            target = rows if name in rows else unknown if name.startswith("tt.") else other
            row = target.setdefault(name, {"cases": [], "merlin_cases": [], "forms": []})
            if case["id"] not in row["cases"]:
                row["cases"].append(case["id"])
            if case["merlin_host"] and case["id"] not in row["merlin_cases"]:
                row["merlin_cases"].append(case["id"])
            form = {k: op.get(k)
                    for k in ("input_types", "output_types", "attributes", "properties")}
            form["attribute_evidence"] = ("recorded-source-dictionaries" if all(isinstance(op.get(k),dict) for k in ("attributes","properties"))
                                          else "not-exposed-by-source-inventory; consult-exact-source")
            digest = hashlib.sha256(json.dumps(form, sort_keys=True).encode()).hexdigest()
            if digest not in forms[name]:
                forms[name][digest] = {"sha256": digest, **form, "observations": []}
                row["forms"].append(forms[name][digest])
            forms[name][digest]["observations"].append({"case": case["id"], "source_id": op["id"]})
    for row in (*rows.values(), *unknown.values(), *other.values()):
        row["status"] = "observed-compositions" if row["cases"] else "unmeasured"
        row["universal_coverage"] = "not-established"
    return {"ttir": rows, "unknown_ttir": unknown, "other_operations": other}


def package_coverage(bundles, *, qualifications=()):
    """Reopen selected receipts before indexing their byte-bound source traces.

    Only explicit paths are read. Standalone Merlin qualifications are matched
    by the exact source receipt digest, never by their source_directory hint.
    """
    if len(bundles) > 64 or len(qualifications) > 64:
        raise ValueError("coverage accepts at most 64 packages/qualifications")
    qualified, qualification_records = {}, []
    for path in qualifications:
        root = Path(path).absolute()
        _receipt(root, "qualification.json")
        receipt = verify_kernel_qualification(root)
        digest = receipt["source_receipt"]["sha256"]
        if digest in qualified:
            raise ValueError("duplicate standalone source qualification")
        qualified[digest] = root
        qualification_records.append({"path": str(root), **_record(root / "qualification.json")})
    cases, packages, selected = [], [], set()
    for path in bundles:
        root = Path(path).absolute()
        receipt = _receipt(root, "capture_receipt.json")
        if receipt.get("schema") not in ("t2m.capture-receipt.v2", "t2m.bound-model-receipt.v1", "t2m.effect-kernel-receipt.v1"):
            raise ValueError("coverage requires shared tensor or bound model packages")
        verify_bundle(root)
        pin = _record(root / "capture_receipt.json")
        if pin["sha256"] in selected:
            raise ValueError("duplicate package receipt in coverage selection")
        selected.add(pin["sha256"])
        model = receipt["schema"] == "t2m.bound-model-receipt.v1"
        effect = receipt["schema"] == "t2m.effect-kernel-receipt.v1"
        effect_reference = _json(root / "reference.json") if effect else None
        mixed = effect and effect_reference["schema"] == "t2m.effect-reference.v2"
        reference_kind = "observable-effects-and-tensors" if mixed else "observable-effects" if effect else "numeric-tensors"
        prefix = "consumer/" if model else ""
        consumer = root if model else qualified.get(pin["sha256"])
        host = None
        if consumer is not None:
            filename = prefix + "host/execution.json"
            if not model or filename in receipt["artifacts"]:
                host = _json(_member(consumer, filename))
        host_schema = "t2m.merlin-effect-host-qualification.v1" if effect else "t2m.merlin-host-qualification.v1" if model else "t2m.merlin-kernel-host-qualification.v1"
        if host is not None and (host.get("schema") != host_schema
                or host.get("host_execution") != ("matched-expected-outcome" if effect else "passed") or host.get("reference_comparison") != "passed"
                or host.get("whole_body_lowering") != "passed"
                or host.get("compiler_input_staging") != "passed"
                or host.get("evaluator_data_supplied_to_compiler") is not False):
            raise ValueError("coverage Merlin observations require the complete host qualification")
        package = {"path": str(root), "receipt": pin, "schema": receipt["schema"],
            "merlin_host": host is not None, "native_reference_comparison": "passed",
            "reference_kind":reference_kind,
            "accelerator_execution": "not-qualified-by-this-report"}
        if mixed:
            package["native_numeric_policy"] = {"results":effect_reference["results"]["comparisons"]}
        if host is not None:
            package["host_report"] = _record(consumer / (prefix + "host/execution.json"))
            if effect:
                package["merlin_effect_policy"] = {"termination":host["termination"],"comparison":"exact-streams-and-termination"}
                if mixed:
                    package["merlin_numeric_policy"] = {"results":host["result_comparisons"]}
            else:
                package["merlin_numeric_policy"] = {key:host[key] for key in ("rtol", "atol")}
        packages.append(package)
        directories = sorted(receipt["kernel_artifacts"].values()) if model else [""]
        for directory in directories:
            base = _member(root, directory) if directory else root
            if any((directory + "/" if directory else "") + name not in receipt["artifacts"]
                   for name in ("provenance.json", "source.ttir", "model.mlir")):
                raise ValueError("source trace/payload is outside the package receipt")
            trace = _json(base / "provenance.json")
            if (trace.get("schema") not in ("t2m.shared-tensor-provenance.v1", "t2m.shared-tensor-provenance.v2")
                    or trace.get("source_sha256") != _record(base / "source.ttir")["sha256"]
                    or trace.get("mlir_sha256") != _record(base / "model.mlir")["sha256"]):
                raise ValueError("coverage source trace must bind its actual source/payload")
            operations = trace["source_operations"]
            source_graph, source_sha256 = "original-source-operations", trace["source_sha256"]
            allocation = trace.get("source_argument_binding", {})
            capsule = allocation.get("source_capsule", {})
            original_inventory = {}
            if capsule:
                source_graph = "allocation-bound-source-operations"
                if capsule["normalized"]:
                    # verify_bundle has checked both receipt and capsule byte links.
                    # Native pre-wrapper inventory stays distinct from bound operations.
                    operations = _json(base / "allocation-inventory.json")["operations"]
                    source_graph = "pre-wrapper-original-source-operations"
                    source_sha256 = _record(base / "allocation-original.ttir")["sha256"]
                    original_inventory = {"original_inventory": _record(base / "allocation-inventory.json"),
                        "bound_source_sha256": trace["source_sha256"],
                        "allocation_bound_operation_counts": dict(Counter(op["opcode"] for op in trace["source_operations"]))}
            cases.append({"id": pin["sha256"] + ":" + (directory or "kernel"), "package_receipt": pin,
                "trace": _record(base / "provenance.json"), "source_sha256": source_sha256,
                "payload_sha256": trace["mlir_sha256"], "source_graph": source_graph, **original_inventory,
                "inventory_authority": "byte-bound producer declaration; not independent source proof",
                "merlin_host": host is not None, "profile": trace.get("profile", "functional-tensor"),
                "reference_kind":reference_kind,
                **({"host_outcome":host["termination"]} if effect and host is not None else {}),
                "abi": trace["spec"], "operations": operations,
                "normalized_operation_counts": dict(Counter(op["opcode"] for op in trace.get("normalized_operations", [])))})
    if set(qualified) - selected:
        raise ValueError("standalone qualification has no matching selected source receipt")
    observations = operation_observations(cases)
    return {"schema": "t2m.package-coverage.v1", "definitions": SOURCE, "definition_count": len(OPERATIONS),
        "selection": "explicit packages only; not an exhaustive kernel corpus",
        "packages": packages, "qualifications": qualification_records,
        "kernel_cases": [{k:v for k,v in case.items() if k != "operations"} for case in cases],
        **observations,
        "observed_definition_count": sum(bool(row["cases"]) for row in observations["ttir"].values()),
        "merlin_observed_definition_count": sum(bool(row["merlin_cases"]) for row in observations["ttir"].values()),
        "unmeasured_definitions": [name for name,row in observations["ttir"].items() if not row["cases"]],
        "refusal_coverage": "not-collected-by-package-selection; use upstream corpus reports separately",
        "operation_execution_coverage": "not-measured; host evidence concerns the containing package and its explicit numeric/effect reference policy",
        "universal_opcode_type_attribute_coverage": "not-established",
        "target_materialization": "not-qualified-by-this-report"}
