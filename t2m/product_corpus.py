"""Measure real kernel packages and refusals against the complete TTIR inventory.

The manifest supplies source, ABI and independent references, never compiler
decisions. Every selected case remains in the report, including invalid source,
located refusals, compilation failures and cases not started within the budget.
"""

import json
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

from .bundle import _json, _record, verify_bundle
from .diagnostics import ConversionError
from .frontend import inspect_ttir
from .package_coverage import _member, operation_observations
from .shared_tensor import TensorABI, convert_shared
from .ttir_definitions import OPERATIONS, SOURCE
from .upstream import inspect_optimizer

OPTIONS = frozenset(("memory_profile", "address_policy", "effect_profile", "f32_atomic_add",
                     "fp8_overflow", "byte_order"))
STAGES = ("source", "conversion", "native_package", "merlin_host", "merlin_replays")


def _implementation():
    root = Path(__file__).parent
    return {"schema": "t2m.product-corpus-implementation.v1", "python": sys.version,
            "sources": {str(path.relative_to(root)): _record(path) for path in sorted(root.rglob("*.py"))}}


def _copy(root, relative, target, artifacts):
    path = _member(root, relative)
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("corpus inputs must be files of at most 32 MiB")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    artifacts[target] = _record(target)
    return target


def _archive(path, prefix, count):
    with zipfile.ZipFile(path) as saved:
        if len(saved.infolist()) > 256 or sum(i.file_size for i in saved.infolist()) > 128 * 1024 * 1024:
            raise ValueError("corpus array archive exceeds its expansion budget")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {f"{prefix}{i}" for i in range(count)}:
            raise ValueError("corpus archives require every argument/result in ABI order")
        return [saved[f"{prefix}{i}"].copy() for i in range(count)]


def _configuration(root, declaration, target, artifacts):
    if isinstance(declaration, str):
        return json.loads(_copy(root, declaration, target, artifacts).read_text())
    if not isinstance(declaration, dict):
        raise TypeError("corpus configuration must be an object or local JSON path")
    _json(target, declaration)
    artifacts[target] = _record(target)
    return declaration


def _case(root, declaration, work, *, optimizer, python, merlin_root, merlin_python, clang):
    stages = dict.fromkeys(STAGES, "not-performed")
    row = {"id": declaration["id"], "stages": stages, "operations": [], "merlin_host": False,
           "native_reference_examples": 0, "merlin_reference_examples": 0,
           "target_materialization": "not-performed", "universal_coverage": "not-established"}
    artifacts, stage = {}, "case_inputs"
    try:
        source = _copy(root, declaration["source"], work / "input/source.ttir", artifacts)
        stage = "source"
        inventory = inspect_ttir(source, work / "inspection")
        row["operations"] = inventory["operations"]
        artifacts[work / "inspection/inventory.json"] = _record(work / "inspection/inventory.json")
        stages[stage] = "passed"
        stage = "case_inputs"
        abi = TensorABI.from_dict(_configuration(root, declaration["abi"], work / "input/abi.json", artifacts))
        options = declaration.get("options", {})
        if not isinstance(options, dict) or set(options) - OPTIONS:
            raise ValueError("unknown corpus compiler options")
        row["options"] = options
        rtol, atol = declaration.get("rtol", 0), declaration.get("atol", 0)
        if not all(type(v) in (int, float) and np.isfinite(v) and v >= 0 for v in (rtol, atol)):
            raise ValueError("corpus comparison requires finite nonnegative tolerances")
        row["numeric_policy"] = {"rtol": rtol, "atol": atol}
        stage = "conversion"
        if "allocation_bindings" in declaration:
            from .allocation_bindings import arguments_from_dict, convert_allocations
            allocation = _configuration(root, declaration["allocation_bindings"], work / "input/allocation.json", artifacts)
            entry, arguments = arguments_from_dict(allocation)
            result = convert_allocations(source, abi, arguments, work / "conversion", source_entry=entry,
                                         optimizer=optimizer, **options)
        else:
            result = convert_shared(source, abi, work / "conversion", optimizer=optimizer, **options)
        stages[stage] = "passed"
        _json(work / "conversion/provenance.json", result.trace)
        (work / "conversion/model.mlir").write_text(result.mlir_text)
        for name in ("provenance.json", "model.mlir"):
            artifacts[work / "conversion" / name] = _record(work / "conversion" / name)
        examples = declaration.get("examples", [])
        if not isinstance(examples, list) or len(examples) > 16:
            raise ValueError("corpus cases require at most sixteen explicit examples")
        if not examples:
            row["status"] = "converted-without-execution-reference"
            return row
        paths = []
        stage = "case_inputs"
        effects = declaration.get("reference_kind", "tensors") == "effects"
        if declaration.get("reference_kind", "tensors") not in ("tensors", "effects"):
            raise ValueError("corpus references must explicitly declare tensors or effects")
        if effects and (rtol or atol):
            raise ValueError("effect cases use explicit outcome/result policies")
        for index, example in enumerate(examples):
            base = work / f"input/example-{index:03d}"
            inputs = _copy(root, example["inputs"], base / "inputs.npz", artifacts)
            reference = _copy(root, example["reference"], base / ("reference.json" if effects else "reference.npz"), artifacts)
            if effects and json.loads(reference.read_text()).get("schema") == "t2m.effect-reference.v2":
                _copy(root, example["results"], base / "reference.npz", artifacts)
            paths.append((inputs, reference))
        stage = "native_package"
        inputs, reference = paths[0]
        arrays = _archive(inputs, "in", len(result.spec.inputs))
        if effects:
            from .effect_bundle import write_effect_bundle
            from .effect_results import load_archive
            expected = json.loads(reference.read_text())
            numeric = load_archive(reference.parent / "reference.npz", len(result.spec.outputs)) if expected.get("schema") == "t2m.effect-reference.v2" else None
            write_effect_bundle(result, work / "package", arrays, expected, work=work / "native", python=python,
                                references=numeric)
        else:
            from .shared_bundle import write_shared_bundle
            expected = _archive(reference, "out", len(result.spec.outputs))
            write_shared_bundle(result, work / "package", arrays, expected, work=work / "native", python=python,
                                rtol=rtol, atol=atol)
        verify_bundle(work / "package")
        artifacts[work / "package/capture_receipt.json"] = _record(work / "package/capture_receipt.json")
        stages[stage], row["native_reference_examples"] = "passed", 1
        if merlin_root is None:
            row["status"] = "native-package-only"
            return row
        stage = "merlin_host"
        from .kernel_consumer import qualify_kernel, verify_kernel_qualification
        qualify_kernel(work / "package", work / "merlin", merlin_root=merlin_root, python=merlin_python,
                       compiler_python=python, clang=clang, rtol=rtol, atol=atol)
        verify_kernel_qualification(work / "merlin")
        artifacts[work / "merlin/qualification.json"] = _record(work / "merlin/qualification.json")
        stages[stage], row["merlin_reference_examples"] = "passed", 1
        stage = "merlin_replays"
        from .invocation import invoke_kernel, verify_invocation
        for index, (inputs, reference) in enumerate(paths[1:], 1):
            target = work / f"invoke-{index:03d}"
            invoke_kernel(work / "merlin", inputs, reference, target, merlin_root=merlin_root,
                          python=merlin_python, rtol=rtol, atol=atol)
            verify_invocation(target)
            artifacts[target / "invocation.json"] = _record(target / "invocation.json")
            row["merlin_reference_examples"] += 1
        stages[stage] = "passed"
        row.update(status="merlin-host-qualified", merlin_host=True)
    except Exception as exc:  # noqa: BLE001 -- Unexpected compiler errors remain failed corpus cases.
        refused = stage == "conversion" and isinstance(exc, ConversionError)
        stages[stage] = "refused" if refused else "failed"
        row.update(status="refused" if refused else "failed", failure={"stage": stage,
            "type": type(exc).__name__, "message": str(exc)[-4000:],
            "opcode": getattr(exc, "opcode", None), "source_id": getattr(exc, "source_id", None)})
        (work / "failure.log").write_text(traceback.format_exc())
        artifacts[work / "failure.log"] = _record(work / "failure.log")
    finally:
        for name in ("source.ttir", "stdout.log", "stderr.log"):
            path = work / "inspection" / name
            if path.is_file():
                artifacts[path] = _record(path)
        row["artifacts"] = {str(path.relative_to(work)): pin for path, pin in artifacts.items()}
        for path, pin in artifacts.items():
            if _record(path) != pin:
                raise ValueError("corpus evidence changed during measurement: " + str(path))
    return row


def _observations(records):
    observations = operation_observations(records)
    names = {r["id"]: {op["opcode"] for op in r["operations"]} for r in records}
    for name, operation in observations["ttir"].items():
        selected = [r for r in records if name in names[r["id"]]]
        operation["stage_cases"] = {stage: [r["id"] for r in selected if r.get("stages", {}).get(stage) == "passed"]
                                    for stage in STAGES}
        operation["attributed_refusals"] = [r["id"] for r in selected if r.get("failure", {}).get("opcode") == name
                                           and r["status"] == "refused"]
    return observations


def qualify_product_corpus(manifest, out, *, optimizer=None, python=None, merlin_root=None,
                           merlin_python=None, clang=None, limit=256, budget=300):
    """Run selected cases serially; success, refusal and unmeasured scope coexist."""
    manifest, out = Path(manifest).absolute(), Path(out).absolute()
    if not 1 <= limit <= 256 or not 0 < budget <= 600:
        raise ValueError("product corpus requires 1..256 cases and at most 600 seconds")
    if any(p.is_symlink() for p in (manifest, *manifest.parents)) or manifest.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("product corpus requires a local nonsymlink bounded manifest")
    configuration = json.loads(manifest.read_text())
    cases = configuration.get("cases")
    if configuration.get("schema") != "t2m.product-corpus.v1" or not isinstance(cases, list) or not 1 <= len(cases) <= 256:
        raise ValueError("expected a t2m.product-corpus.v1 manifest with 1..256 cases")
    ids = [case.get("id") if isinstance(case, dict) else None for case in cases]
    if any(not isinstance(id_, str) or not 1 <= len(id_) <= 96 or
           any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in id_) for id_ in ids) or len(set(ids)) != len(ids):
        raise ValueError("corpus case IDs must be unique local names")
    tool = inspect_optimizer(optimizer)
    out.mkdir(parents=True, exist_ok=False)
    (out / "manifest.json").write_bytes(manifest.read_bytes())
    implementation = _implementation()
    _json(out / "implementation.json", implementation)
    started, records = time.monotonic(), []
    for index, declaration in enumerate(cases):
        work = out / declaration["id"]
        work.mkdir()
        if index >= limit or time.monotonic() - started >= budget:
            row = {"id": declaration["id"], "status": "not-started", "merlin_host": False,
                   "operations": [], "reason": "limit" if index >= limit else "budget"}
        else:
            row = _case(manifest.parent, declaration, work, optimizer=tool, python=python,
                        merlin_root=merlin_root, merlin_python=merlin_python, clang=clang)
        if _implementation() != implementation:
            raise ValueError("corpus compiler implementation changed during measurement")
        _json(work / "case.json", row)
        records.append(row)
        observations = _observations(records)
        report = {"schema": "t2m.product-corpus-report.v1", "definitions": SOURCE, "definition_count": len(OPERATIONS),
            "manifest": _record(out / "manifest.json"), "implementation": _record(out / "implementation.json"), "optimizer": tool.record,
            "selected_cases": len(cases), "recorded_cases": len(records),
            "measured_cases": sum(r["status"] != "not-started" for r in records),
            "complete_selection_measured": len(records) == len(cases) and all(r["status"] != "not-started" for r in records),
            "all_selected_cases_merlin_host_qualified": len(records) == len(cases) and all(r["merlin_host"] for r in records),
            "unmeasured_definitions": [name for name, op in observations["ttir"].items() if not op["cases"]],
            "stage_definition_counts": {stage: sum(bool(op["stage_cases"][stage]) for op in observations["ttir"].values()) for stage in STAGES},
            "case_records": {r["id"]: _record(out / r["id"] / "case.json") for r in records},
            "cases": records, **observations, "target_materialization": "not-performed",
            "source_equivalence": "not-established", "universal_opcode_type_attribute_coverage": "not-established",
            "interpretation": "Original-source composition observations; failed/refused and unstarted cases stay in the selection. Native compares the first example; Merlin additionally replays each remaining example. No per-operation execution or universal support claim.",
            "elapsed_seconds": round(time.monotonic() - started, 3)}
        _json(out / "coverage.json", report)
    return report


def verify_product_corpus(out):
    """Reopen source custody and actual package/library evidence; run no kernels."""
    out = Path(out).absolute()
    if any(path.is_symlink() for path in (out, *out.parents)):
        raise ValueError("corpus evidence requires local nonsymlink paths")
    report = json.loads(_member(out, "coverage.json").read_text())
    if report.get("schema") != "t2m.product-corpus-report.v1" or report.get("definitions") != SOURCE:
        raise ValueError("unknown product corpus report or source definitions")
    if _record(_member(out, "manifest.json")) != report["manifest"]:
        raise ValueError("corpus manifest integrity mismatch")
    if _record(_member(out, "implementation.json")) != report["implementation"]:
        raise ValueError("corpus compiler implementation integrity mismatch")
    for row in report["cases"]:
        case = _member(out, row["id"] + "/case.json")
        if _record(case) != report["case_records"].get(row["id"]) or json.loads(case.read_text()) != row:
            raise ValueError("corpus case record integrity mismatch")
        work = case.parent
        for name, pin in row.get("artifacts", {}).items():
            if _record(_member(work, name)) != pin:
                raise ValueError("corpus artifact integrity mismatch: " + name)
        stages = row.get("stages", {})
        if stages.get("source") == "passed":
            inventory = json.loads(_member(work, "inspection/inventory.json").read_text())
            if (inventory.get("schema") != "t2m.native-source-inventory.v1"
                    or inventory.get("normalization") != "not-performed" or inventory["operations"] != row["operations"]
                    or _record(_member(work, "inspection/source.ttir")) != _record(_member(work, "input/source.ttir"))):
                raise ValueError("corpus original-source inventory mismatch")
        if stages.get("native_package") == "passed":
            verify_bundle(work / "package")
        if stages.get("merlin_host") == "passed":
            from .kernel_consumer import verify_kernel_qualification
            verify_kernel_qualification(work / "merlin")
        for name in row.get("artifacts", {}):
            if name.startswith("invoke-") and name.endswith("/invocation.json"):
                from .invocation import verify_invocation
                verify_invocation(work / Path(name).parent)
        if row["merlin_host"] != all(stages.get(stage) == "passed" for stage in STAGES):
            raise ValueError("corpus host status differs from the complete measured stages")
    observed = _observations(report["cases"])
    for category in ("ttir", "unknown_ttir", "other_operations"):
        if observed[category] != report[category]:
            raise ValueError("corpus operation observations differ from original cases")
    selected = json.loads((out / "manifest.json").read_text())["cases"]
    if (report["selected_cases"] != len(selected) or report["recorded_cases"] != len(report["cases"])
            or report["definition_count"] != len(OPERATIONS)
            or report["measured_cases"] != sum(r["status"] != "not-started" for r in report["cases"])
            or report["unmeasured_definitions"] != [name for name, op in observed["ttir"].items() if not op["cases"]]
            or [r["id"] for r in report["cases"]] != [r["id"] for r in selected][:len(report["cases"])]):
        raise ValueError("corpus case denominator differs from the manifest")
    complete = len(report["cases"]) == len(selected)
    if (report["complete_selection_measured"] != (complete and all(r["status"] != "not-started" for r in report["cases"]))
            or report["all_selected_cases_merlin_host_qualified"] != (complete and all(r["merlin_host"] for r in report["cases"]))
            or report["stage_definition_counts"] != {stage: sum(bool(op["stage_cases"][stage]) for op in observed["ttir"].values()) for stage in STAGES}):
        raise ValueError("corpus aggregate status differs from actual case evidence")
    return report
