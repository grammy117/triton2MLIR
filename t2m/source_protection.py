"""Make source numeric contracts executable before allocation normalization."""

import json
from io import StringIO
from pathlib import Path

from xdsl.printer import Printer

from .diagnostics import ConversionError
from .source_capsule import record

SCHEMA = "t2m.source-protection.v1"
FILES = frozenset(
    {
        "allocation-generic.mlir",
        "allocation-native-print.json",
        "allocation-protected.ttir",
        "allocation-protection.json",
    }
)


def protect(
    source,
    work,
    *,
    source_entry,
    fp8_overflow=None,
    effect_profile="ordinary",
    f32_atomic_add=None,
    assembly_provider=None,
):
    """Audit the original source without granting its eventual wrapper ABI."""
    from .fp8_casts import expand_fp8_casts, validate_fp8_casts
    from .frontend import read_ttir_generic
    from .shared_tensor import _preflight, _source_ids, parse_shared
    from .source_graph import inventory

    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    original = Path(source).read_bytes()
    native = read_ttir_generic(source, work / "native")
    if native["source_sha256"] != record(original)["sha256"]:
        raise ConversionError("original allocation source changed before protection")
    generic = (work / "native/lowered.mlir").read_text()
    module = parse_shared(generic)
    records = _preflight(
        module,
        source_entry=source_entry,
        atomics=effect_profile in ("atomics", "typed-atomics"),
        typed_atomics=effect_profile == "typed-atomics",
        f32_atomic_add=f32_atomic_add,
        allow_composite_dots=True,
        assembly_provider=assembly_provider,
    )
    validate_fp8_casts(module, _source_ids, overflow=fp8_overflow)
    graph = inventory(module, records)
    casts = expand_fp8_casts(
        module, _source_ids, overflow=fp8_overflow, tensor_carrier="arith"
    )
    assembly = []
    if assembly_provider is not None:
        from .source_assembly import expand_source_assembly

        assembly = expand_source_assembly(
            module, _source_ids, provider=assembly_provider, tensor_carrier="arith"
        )
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(
        module
    )
    protected = stream.getvalue() + "\n"
    report = {
        "schema": SCHEMA,
        "source_entry": source_entry,
        "fp8_overflow": fp8_overflow,
        "artifacts": {
            "original": record(original),
            "generic": record(generic),
            "protected": record(protected),
        },
        "source_graph": graph,
        "fp8_conversions": casts,
        "assembly_conversions": assembly,
        "grants_semantics": False,
    }
    artifacts = {
        "allocation-generic.mlir": generic,
        "allocation-native-print.json": json.dumps(native, indent=2) + "\n",
        "allocation-protected.ttir": protected,
        "allocation-protection.json": json.dumps(report, indent=2) + "\n",
    }
    for name, data in artifacts.items():
        (work / name).write_text(data)
    validate(artifacts, original, source_entry)
    return work / "allocation-protected.ttir", artifacts, report


def validate(artifacts, original, source_entry):
    """Check compiler-byte custody, not arithmetic or source equivalence."""
    report = json.loads(artifacts["allocation-protection.json"])
    native = json.loads(artifacts["allocation-native-print.json"])
    if (
        set(report)
        != {
            "schema",
            "source_entry",
            "fp8_overflow",
            "artifacts",
            "source_graph",
            "fp8_conversions",
            "assembly_conversions",
            "grants_semantics",
        }
        or report.get("schema") != SCHEMA
        or report.get("source_entry") != source_entry
        or report.get("grants_semantics") is not False
        or report.get("fp8_overflow")
        not in (None, "ieee", "satfinite", "satfinite-nan-inf", "per-cast")
        or report.get("artifacts")
        != {
            "original": record(original),
            "generic": record(artifacts["allocation-generic.mlir"]),
            "protected": record(artifacts["allocation-protected.ttir"]),
        }
        or not isinstance(report.get("source_graph"), dict)
        or set(report["source_graph"]) != {"operations", "blocks", "attributes"}
        or not isinstance(report["source_graph"]["operations"], list)
        or not report["source_graph"]["operations"]
        or not isinstance(report["source_graph"]["blocks"], list)
        or not isinstance(report.get("fp8_conversions"), list)
        or not isinstance(report.get("assembly_conversions"), list)
    ):
        raise ValueError(
            "allocation source protection differs from its original/generic/protected custody"
        )
    if (
        set(native)
        != {
            "schema",
            "triton_version",
            "flags",
            "operations",
            "normalization",
            "target_backend",
            "native_source_verified",
            "native_generic_roundtrip_verified",
            "grants_semantics",
            "source_sha256",
            "output_sha256",
            "implementation",
        }
        or native.get("schema") != "t2m.native-source-print.v1"
        or native.get("normalization") != "not-performed"
        or native.get("target_backend") != "not-invoked"
        or native.get("grants_semantics") is not False
        or native.get("native_source_verified") is not True
        or native.get("native_generic_roundtrip_verified") is not True
        or native.get("flags")
        != [
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
            "--mlir-print-debuginfo",
        ]
        or native.get("source_sha256") != record(original)["sha256"]
        or native.get("output_sha256")
        != record(artifacts["allocation-generic.mlir"])["sha256"]
    ):
        raise ValueError(
            "allocation source protection lacks its complete native source round trip"
        )
    implementation = native.get("implementation", {})
    digest = implementation.get("binary_sha256")
    if (
        set(implementation)
        != {
            "binary_sha256",
            "binary_bytes",
            "printer_options",
            "source_reference_revision",
            "source_reference_is_build_receipt",
        }
        or type(implementation.get("binary_bytes")) is not int
        or implementation["binary_bytes"] <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or implementation.get("source_reference_is_build_receipt") is not False
    ):
        raise ValueError(
            "allocation source protection lacks the native printer implementation identity"
        )
    return report
