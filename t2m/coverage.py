"""Measure a fixed upstream pipeline on a real corpus, with honest denominators.

This is pass/corpus qualification, not functional-ABI or numeric coverage. It
does not execute RUN comments or use fixture names to choose compiler behavior.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .diagnostics import ConversionError
from .shared_tensor import opcode, parse_shared
from .upstream import inspect_optimizer, lower_with_shared


def chunks(source):
    """Split the standard MLIR test delimiter, not IR syntax or RUN commands."""
    lines = []
    for line in source.splitlines(keepends=True):
        if line.startswith("// ") and line[3:].strip() and set(line[3:].strip()) == {"-"}:
            if "".join(lines).strip():
                yield "".join(lines)
            lines = []
        else:
            lines.append(line)
    if "".join(lines).strip():
        yield "".join(lines)


def inventory(source):
    module = parse_shared(source)
    return Counter(opcode(op) for op in module.walk())


def qualify_corpus(root, out, *, optimizer=None, stage="tensor-ir", limit=1000, budget=120):
    if not 1 <= limit <= 2000 or not 0 < budget <= 600:
        raise ValueError("corpus qualification needs 1..2000 cases and at most 600 seconds")
    root, out = Path(root).resolve(), Path(out)
    if not root.is_dir():
        raise ValueError("corpus root must exist")
    paths = sorted(p for p in root.rglob("*.mlir") if p.is_file() and not p.is_symlink())
    cases = [(p, i, chunk) for p in paths for i, chunk in enumerate(chunks(p.read_text()))]
    if not cases:
        raise ValueError("no MLIR corpus cases found")
    tool = inspect_optimizer(optimizer)
    out.mkdir(parents=True, exist_ok=False)
    start, records, coverage = time.monotonic(), [], defaultdict(Counter)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True, timeout=5, check=True).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        revision = None
    for path, index, source in cases[:limit]:
        if time.monotonic() - start >= budget:
            break
        case_dir = out / f"case-{len(records):04d}"
        case_dir.mkdir()
        saved = case_dir / "input.ttir"
        saved.write_text(source)
        record = {"path": str(path.relative_to(root)), "chunk": index,
                  "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "source_native_verified": False, "stage_completed": False,
                  "functional_tensor_qualified": False, "numeric_equivalence": "not-tested"}
        try:
            lower_with_shared(saved, case_dir / "source", optimizer=tool, stage="source", timeout=min(10, budget))
            record["source_native_verified"] = True
            try:
                source_ops = inventory((case_dir / "source/lowered.mlir").read_text())
                record["source_ops"] = dict(source_ops)
            except Exception as exc:
                record["inventory_error"] = f"{type(exc).__name__}: {exc}"[:1000]
            if stage in ("tensor-ir", "source-word-tensor"):
                lower_with_shared(saved, case_dir / "inline", optimizer=tool, stage="inline", timeout=min(10, budget))
                saved = case_dir / "inline/lowered.mlir"
            lower_with_shared(saved, case_dir / "stage", optimizer=tool, stage=stage, timeout=min(10, budget))
            record["stage_completed"] = True
            try:
                emitted = inventory((case_dir / "stage/lowered.mlir").read_text())
                record["emitted_ops"] = dict(emitted)
                record["remaining_project_ops"] = {name: count for name, count in emitted.items()
                    if name.split(".")[0] in ("tt", "tts", "tptr", "ttx", "ttg", "ttng", "gpu", "ptr")}
            except Exception as exc:
                record["output_inventory_error"] = f"{type(exc).__name__}: {exc}"[:1000]
        except ConversionError as exc:
            record["diagnostic"] = str(exc)[-2000:]
        for name in record.get("source_ops", {}):
            coverage[name]["cases_observed"] += 1
            coverage[name]["stage_success_cases"] += int(record["stage_completed"])
            coverage[name]["stage_failure_cases"] += int(not record["stage_completed"])
        records.append(record)
        # Keep an inspectable report even if a later fixture or budget stops us.
        (case_dir / "case.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    report = {"schema": "t2m.upstream-corpus-coverage.v1", "root": str(root), "revision": revision,
              "stage": stage, "optimizer": {"path": str(tool.path), **tool.record},
              "discovered_files": len(paths), "discovered_cases": len(cases), "measured_cases": len(records),
              "complete_corpus_measured": len(records) == len(cases),
              "source_native_verified": sum(r["source_native_verified"] for r in records),
              "stage_successes": sum(r["stage_completed"] for r in records),
              "observed_ttir_cases": sum("tt.func" in r.get("source_ops", {}) for r in records),
              "observed_ttir_stage_successes": sum(r["stage_completed"] and "tt.func" in r.get("source_ops", {}) for r in records),
              "inventory_failures": sum("inventory_error" in r for r in records),
              "elapsed_seconds": round(time.monotonic() - start, 3),
              "operation_observations": {n: dict(c) for n, c in sorted(coverage.items())},
              "functional_tensor_coverage": "not-measured-by-this-corpus-driver",
              "numeric_equivalence": "not-tested", "consumer_execution": "not-performed",
              "interpretation": "Each opcode occurs in compositions; success is not universal opcode/type/attribute coverage. Negative fixtures and stage-specific inputs are retained in the denominator.",
              "cases": records}
    (out / "coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
