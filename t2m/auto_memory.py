"""Try supported portable representations using source IR and the ABI only.

All ordinary conversions and source/memory audits remain authoritative.
Attempts are bounded and sequential. No evaluator, target performance model,
kernel-name rule or numeric/effect-policy change selects a representation.
"""

import hashlib
import json
from pathlib import Path

from .diagnostics import ConversionError
from .memory_selection import SCHEMA, SCOPE, STRATEGY, candidates, validate_selection


def convert(
    source,
    abi,
    work,
    *,
    optimizer,
    address_policy,
    effect_profile,
    f32_atomic_add,
    fp8_overflow,
    byte_order,
    assembly_provider,
    source_layout,
):
    from .shared_tensor import convert_shared

    requested = {
        "address_policy": address_policy,
        "effect_profile": effect_profile,
        "f32_atomic_add": f32_atomic_add,
        "fp8_overflow": fp8_overflow,
        "byte_order": byte_order,
    }
    sequence = candidates(requested)
    source, work = Path(source), Path(work)
    original = source.read_bytes()
    work.mkdir(parents=True, exist_ok=False)
    snapshot = work / "source.ttir"
    snapshot.write_bytes(original)
    record = {
        "schema": SCHEMA,
        "strategy": STRATEGY,
        "scope": SCOPE,
        "requested": requested,
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "attempts": [],
        "status": "not-admitted",
    }

    def save():
        (work / "memory-selection.json").write_text(
            json.dumps(record, sort_keys=True, indent=2) + "\n"
        )

    last_error = None
    for ordinal, (profile, policy) in enumerate(sequence):
        if source.read_bytes() != original or snapshot.read_bytes() != original:
            save()
            raise ConversionError("source changed during automatic memory selection")
        attempt = {
            "memory_profile": profile,
            "address_policy": policy,
            "status": "not-admitted",
        }
        try:
            result = convert_shared(
                snapshot,
                abi,
                work / f"attempt-{ordinal}",
                optimizer=optimizer,
                memory_profile=profile,
                address_policy=policy,
                effect_profile=effect_profile,
                f32_atomic_add=f32_atomic_add,
                fp8_overflow=fp8_overflow,
                byte_order=byte_order,
                assembly_provider=assembly_provider,
                source_layout=source_layout,
            )
        except ConversionError as error:
            attempt.update(
                diagnostic=str(error)[:4096],
                source_id=error.source_id,
                opcode=error.opcode,
            )
            record["attempts"].append(attempt)
            save()
            last_error = error
            continue
        except Exception as error:
            attempt.update(
                status="failed",
                diagnostic=str(error)[:4096],
                exception=type(error).__name__,
            )
            record["attempts"].append(attempt)
            save()
            raise
        if source.read_bytes() != original or snapshot.read_bytes() != original:
            save()
            raise ConversionError("source changed during automatic memory selection")
        attempt["status"] = "admitted"
        record["attempts"].append(attempt)
        record.update(status="admitted", source_sha256=result.trace["source_sha256"])
        result.trace["memory_selection"] = record
        validate_selection(result.trace)
        save()
        return result
    assert last_error is not None
    raise ConversionError(
        "no portable memory representation was admitted; see memory-selection.json",
        source_id=last_error.source_id,
        opcode=last_error.opcode,
    ) from last_error
