"""Reuse LLVM's CFG-to-SCF conversion through a typed TT/function bridge."""

import json
from pathlib import Path

from xdsl.dialects import builtin, func

from .diagnostics import ConversionError
from .source_capsule import record

SCHEMA = "t2m.control-flow-normalization.v1"
FILES = frozenset(
    {
        "cfg-input.mlir",
        "cfg-lifted.mlir",
        "cfg-output.ttir",
        "cfg-source-graph.json",
        "cfg-stage.json",
        "cfg-stderr.log",
    }
)
FLAGS = [
    "--mlir-disable-threading",
    "--verify-each",
    "--mlir-print-op-generic",
    "--mlir-print-debuginfo",
    "--mlir-print-local-scope",
    "--canonicalize",
    "--lift-cf-to-scf",
    "--canonicalize",
]


def required(module):
    from .shared_tensor import opcode

    return any(
        opcode(op) in ("cf.br", "cf.cond_br", "cf.switch")
        or any(len(region.blocks) > 1 for region in op.regions)
        for op in module.walk()
    )


def _functions(module, *, restore=False):
    """Change only equivalent function/call/return interfaces and ownership."""
    from .shared_tensor import opcode

    classes = {
        "tt.func": func.FuncOp,
        "tt.call": func.CallOp,
        "tt.return": func.ReturnOp,
    }
    if restore:
        classes = {
            f"func.{name}": builtin.UnregisteredOp.with_name(f"tt.{name}")
            for name in ("func", "call", "return")
        }
    for op in list(module.walk()):
        cls = classes.get(opcode(op))
        if cls is None:
            continue
        replacement = cls.create(
            operands=op.operands,
            result_types=[v.type for v in op.results],
            regions=[op.detach_region(region) for region in list(op.regions)],
            properties=dict(op.properties),
            attributes={
                key: value
                for key, value in op.attributes.items()
                if key not in ("op_name", "op_name__")
            },
            location=op.location,
        )
        op.parent.insert_op_before(replacement, op)
        for old, new in zip(op.results, replacement.results, strict=True):
            old.replace_all_uses_with(new)
        op.parent.erase_op(op)


def normalize(module, work, *, optimizer, source_sha256, source_graph, entry):
    from .shared_tensor import parse_shared
    from .upstream import lower_with_shared

    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    module = module.clone()
    _functions(module)
    # Locations carry original IDs into the official rewrite and serialized IR.
    from io import StringIO

    from xdsl.printer import Printer

    stream = StringIO()
    Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(
        module
    )
    input_ = stream.getvalue() + "\n"
    source = work / "input.mlir"
    source.write_text(input_)
    stage = lower_with_shared(
        source, work / "native", optimizer=optimizer, stage="control-flow"
    )
    if stage["command"][1:-1] != FLAGS:
        raise ConversionError(
            "CFG native stage differs from its declared fixed pass pipeline"
        )
    lifted = (work / "native/lowered.mlir").read_text()
    module = parse_shared(lifted)
    _functions(module, restore=True)
    if required(module):
        raise ConversionError(
            "official CFG-to-SCF pass retained unstructured control flow"
        )
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(
        module
    )
    output = stream.getvalue() + "\n"
    custody = {
        "schema": "t2m.control-flow-stage.v1",
        "flags": FLAGS,
        "artifacts": stage["artifacts"],
        "stage_completed": stage["stage_completed"],
        "optimizer": {key: stage["optimizer"][key] for key in ("sha256", "bytes")},
        "grants_semantics": False,
    }
    artifacts = {
        "cfg-input.mlir": input_,
        "cfg-lifted.mlir": lifted,
        "cfg-output.ttir": output,
        "cfg-source-graph.json": json.dumps(source_graph, indent=2) + "\n",
        "cfg-stage.json": json.dumps(custody, indent=2) + "\n",
        "cfg-stderr.log": (work / "native/stderr.log").read_text(),
    }
    report = {
        "schema": SCHEMA,
        "entry": entry,
        "source_sha256": source_sha256,
        "artifacts": {name: record(data) for name, data in artifacts.items()},
        "grants_semantics": False,
    }
    return module, artifacts, report


def members(trace):
    report = trace.get("control_flow_normalization")
    if report is None:
        return frozenset()
    if (
        not isinstance(report, dict)
        or set(report)
        != {"schema", "entry", "source_sha256", "artifacts", "grants_semantics"}
        or report.get("schema") != SCHEMA
        or report.get("grants_semantics") is not False
        or set(report.get("artifacts", {})) != FILES
    ):
        raise ValueError(
            "CFG source custody lacks its explicit complete artifact roster"
        )
    return FILES


def validate(trace, artifacts, source):
    names = members(trace)
    if not names:
        return
    report = trace["control_flow_normalization"]
    if (
        report["entry"] != trace.get("spec", {}).get("entry")
        or report["source_sha256"] != record(source)["sha256"]
        or any(
            name not in artifacts
            or report["artifacts"][name] != record(artifacts[name])
            for name in names
        )
    ):
        raise ValueError(
            "CFG source custody differs from original source or retained compiler bytes"
        )
    graph = json.loads(artifacts["cfg-source-graph.json"])
    if (
        graph.get("operations") != trace.get("source_operations")
        or graph.get("blocks") != trace.get("original_blocks")
        or graph.get("attributes") != trace.get("original_attributes")
    ):
        raise ValueError(
            "CFG source graph differs from its complete original block/operation inventory"
        )
    stage = json.loads(artifacts["cfg-stage.json"])
    if (
        set(stage)
        != {
            "schema",
            "flags",
            "artifacts",
            "stage_completed",
            "optimizer",
            "grants_semantics",
        }
        or stage.get("schema") != "t2m.control-flow-stage.v1"
        or stage.get("flags") != FLAGS
        or stage.get("stage_completed") is not True
        or stage.get("grants_semantics") is not False
        or stage.get("artifacts")
        != {
            "source.ttir": record(artifacts["cfg-input.mlir"]),
            "lowered.mlir": record(artifacts["cfg-lifted.mlir"]),
            "stderr.log": record(artifacts["cfg-stderr.log"]),
        }
    ):
        raise ValueError(
            "CFG native stage does not bind its fixed pass and input/output bytes"
        )
    optimizer = stage.get("optimizer", {})
    digest = optimizer.get("sha256")
    if (
        set(optimizer) != {"sha256", "bytes"}
        or type(optimizer.get("bytes")) is not int
        or optimizer["bytes"] <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("CFG native stage lacks its actual optimizer fingerprint")
