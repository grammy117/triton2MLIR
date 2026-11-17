"""Common parent/kernel retention and byte custody for bound model profiles."""

import hashlib
import json
import shutil
from pathlib import Path

from .bundle import _json, _record
from .memory_contract import TENSOR_PROFILE
from .package_compatibility import kernel_representation_differences
from .source_capsule import validate as validate_source_capsule


def retain_parent(capture, parent, out, data_members):
    (out / "parent").mkdir()
    shutil.copyfile(
        capture / "capture_receipt.json", out / "parent/capture_receipt.json"
    )
    bindings = {}
    for name in parent["artifacts"]:
        destination = (
            name if name in data_members and name != "model.mlir" else "parent/" + name
        )
        shutil.copyfile(capture / name, out / destination)
        bindings[name] = destination
    return bindings


def retain_kernels(bindings, trace, out, *, located=False):
    selected = {record["implementation_sha256"] for record in trace["calls"]}
    implementations = {
        binding.implementation.trace["mlir_sha256"]: binding.implementation
        for binding in bindings
        if binding.implementation.trace["mlir_sha256"] in selected
    }
    artifacts, profiles = {}, {}
    for ordinal, (digest, implementation) in enumerate(implementations.items()):
        validate_source_capsule(
            implementation.trace,
            implementation.auxiliary_artifacts,
            implementation.source_ttir,
        )
        if hashlib.sha256(implementation.mlir_text.encode()).hexdigest() != digest:
            raise ValueError("kernel implementation changed during package retention")
        directory = out / "kernels" / str(ordinal)
        directory.mkdir(parents=True)
        (directory / "source.ttir").write_text(implementation.source_ttir)
        (directory / "model.mlir").write_text(implementation.mlir_text)
        _json(directory / "spec.json", implementation.spec.as_dict())
        _json(directory / "provenance.json", implementation.trace)
        for name, payload in implementation.auxiliary_artifacts.items():
            (directory / name).write_text(payload)
        if located:
            from .executed_model_binding import serialize

            source = serialize(implementation.module).encode()
            if trace["located_implementations"][digest] != {
                "sha256": hashlib.sha256(source).hexdigest(),
                "bytes": len(source),
            }:
                raise ValueError("located kernel changed during package retention")
            (directory / "located.mlir").write_bytes(source)
        name = str(directory.relative_to(out))
        artifacts[digest] = name
        profiles[name] = {
            "profile": implementation.trace.get("profile", TENSOR_PROFILE),
            "differences": kernel_representation_differences(
                implementation.trace, dynamic=implementation.spec.dynamic
            ),
        }
    return artifacts, profiles


def checked_members(out, receipt, required):
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not required <= set(artifacts):
        raise ValueError("not a complete bound model receipt")
    for name, expected in artifacts.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or name == "capture_receipt.json"
        ):
            raise ValueError("invalid bound artifact path")
        path = out / relative
        parts = [
            out.joinpath(*relative.parts[:i]) for i in range(len(relative.parts) + 1)
        ]
        if (
            any(part.is_symlink() for part in parts)
            or not path.is_file()
            or _record(path) != expected
        ):
            raise ValueError("bound bundle integrity mismatch: " + name)
    parent_path = out / "parent/capture_receipt.json"
    parent = json.loads(parent_path.read_text())
    if _record(parent_path) != receipt["parent_receipt"] or set(
        parent["artifacts"]
    ) != set(receipt["parent_artifact_bindings"]):
        raise ValueError("incomplete parent receipt binding")
    for old_name, destination in receipt["parent_artifact_bindings"].items():
        if (
            destination not in artifacts
            or _record(out / destination) != parent["artifacts"][old_name]
        ):
            raise ValueError("parent artifact relation failed: " + old_name)
    return parent
