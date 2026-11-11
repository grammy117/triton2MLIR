"""Source layout facts for packed computation; no consumer hardware placement."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from xdsl.dialects.builtin import StringAttr
from xdsl.printer import Printer

from .diagnostics import ConversionError


KEY = "t2m.map_key"


def options(value):
    if (not isinstance(value, dict) or set(value) != {"target", "num_warps", "threads_per_warp", "num_ctas"}
            or not isinstance(value["target"], str) or not 1 <= len(value["target"]) <= 128
            or any(type(value[k]) is not int or not 1 <= value[k] <= 1024 or value[k] & (value[k] - 1)
                   for k in ("num_warps", "threads_per_warp", "num_ctas"))
            or math.prod(value[k] for k in ("num_warps", "threads_per_warp", "num_ctas")) > 65536):
        raise ValueError("source layout needs an explicit target and bounded power-of-two warp/thread/CTA counts")
    return value


def _dimensions(plan):
    dimensions = plan["out_dims"]
    if (not dimensions or len({name for name, _ in dimensions}) != len(dimensions)
            or any(not isinstance(name, str) or type(size) is not int or size < 1 or size & (size - 1) for name, size in dimensions)):
        raise ValueError("invalid linear layout dimensions")
    bases = plan["bases"]
    if len({name for name, _ in bases}) != len(bases):
        raise ValueError("duplicate linear layout input dimensions")
    for _, vectors in bases:
        for vector in vectors:
            if len(vector) != len(dimensions) or any(type(v) is not int or not 0 <= v < size for v, (_, size) in zip(vector, dimensions)):
                raise ValueError("invalid linear layout basis")
    return {name: 1 << len(vectors) for name, vectors in bases}, dict(dimensions)


def apply(plan, inputs):
    result = {name: 0 for name, _ in plan["out_dims"]}
    for name, vectors in plan["bases"]:
        for bit, vector in enumerate(vectors):
            if inputs[name] & (1 << bit):
                for (output, _), value in zip(plan["out_dims"], vector, strict=True):
                    result[output] ^= value
    return result


def validate(row):
    forward, inverse = row["forward"], row["inverse"]
    inputs, outputs = _dimensions(forward)
    inverse_inputs, inverse_outputs = _dimensions(inverse)
    if (list(inputs) != ["register", "lane", "warp", "block"] or outputs != {f"dim{i}": n for i, n in enumerate(row["shape"])}
            or inverse_inputs != outputs or inverse_outputs != inputs
            or type(row["pack"]) is not int or row["pack"] < 2 or row["pack"] & (row["pack"] - 1)
            or inputs["register"] % row["pack"] or math.prod(outputs.values()) >= 1 << 62):
        raise ValueError("source packing and coordinate layout disagree")
    # Check the inverse on a basis of both binary spaces, without enumerating
    # tensor elements or trusting a claimed bijection/provenance annotation.
    for first, second, domain in ((forward, inverse, inputs), (inverse, forward, outputs)):
        for name, size in domain.items():
            for bit in range(size.bit_length() - 1):
                point = {n: (1 << bit) if n == name else 0 for n in domain}
                if apply(second, apply(first, point)) != point:
                    raise ValueError("source layout does not establish an exact coordinate bijection")
    return inputs


def validate_witness(witness):
    from .frontend import PINNED_TRITON
    if (witness.get("schema") != "t2m.source-map-layout.v1" or witness.get("triton_version") != PINNED_TRITON
            or witness.get("pipeline") != ["convert-triton-to-tritongpu"] or witness.get("device_codegen") != "not-invoked"
            or witness.get("source_sha256") != hashlib.sha256(witness["source_ir"].encode()).hexdigest()
            or witness.get("ttgir_sha256") != hashlib.sha256(witness["ttgir"].encode()).hexdigest()
            or witness.get("dependency_source_closure") != "not-established"):
        raise ValueError("source layout witness is incomplete or changed")
    options(witness["source_options"])
    library = witness["native_library"]
    if (not isinstance(library.get("path"), str) or type(library.get("bytes")) is not int or library["bytes"] < 1
            or not isinstance(library.get("sha256"), str) or len(library["sha256"]) != 64):
        raise ValueError("source layout witness lacks its actual native implementation identity")
    if not witness["maps"] or len({row["id"] for row in witness["maps"]}) != len(witness["maps"]):
        raise ValueError("source layout witness map identities are incomplete")
    for row in witness["maps"]:
        validate(row)


def validate_trace(trace):
    witness = trace.get("source_layout_specialization")
    reports = [entry["packing_proof"] for entry in trace.get("transformations", [])
               if entry.get("kind") == "ttir-elementwise-map" and entry.get("packing_proof", {}).get("method") == "source-linear-layout-v1"]
    if witness is None:
        if reports:
            raise ValueError("source-grouped computation lacks its source layout witness")
        return
    validate_witness(witness)
    maps = {row["id"]: row for row in witness["maps"]}
    if len(reports) != len(maps) or {r["source_map"] for r in reports} != set(maps) or any(
            r["layout"] != maps[r["source_map"]]
            or r.get("logical_groups") != math.prod(r["layout"]["shape"]) // r["layout"]["pack"]
            or r.get("group_compute") != "one-original-scalar-region-per-logical-group" for r in reports):
        raise ValueError("source-grouped computation differs from its source specialization")


def prepare(module, work, selection):
    from .shared_tensor import opcode
    selection = options(selection)
    maps = [op for op in module.walk() if opcode(op) == "tt.map_elementwise" and op.properties["pack"].value.data > 1]
    if not maps:
        raise ConversionError("source layout specialization currently requires a packed map", opcode="tt.map_elementwise")
    for i, op in enumerate(maps):
        op.attributes[KEY] = StringAttr(f"source.map.{i}")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    source, record = work / "source.ttir", work / "layout.json"
    with source.open("w") as stream:
        Printer(stream=stream, print_generic_format=True, print_debuginfo=True).print_op(module)
    command = [sys.executable, "-c", "import sys; sys.path.insert(0,sys.argv[1]); from t2m.frontend import _extract_map_layouts; _extract_map_layouts(*sys.argv[2:])",
               str(Path(__file__).resolve().parents[1]), str(source.resolve()), str(record.resolve()), json.dumps(selection)]
    with (work / "stdout.log").open("xb") as stdout, (work / "stderr.log").open("xb") as stderr:
        try:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=30,
                env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(work.resolve())))
        except subprocess.TimeoutExpired as exc:
            raise ConversionError("source layout extraction exceeded 30 seconds", opcode="tt.map_elementwise") from exc
    if process.returncode or not record.is_file():
        raise ConversionError("source layout extraction failed; inspect " + str(work / "stderr.log"), opcode="tt.map_elementwise")
    witness = json.loads(record.read_text())
    if (witness["source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()
            or witness["ttgir_sha256"] != hashlib.sha256(witness["ttgir"].encode()).hexdigest()
            or witness["source_options"] != selection or witness["device_codegen"] != "not-invoked"):
        raise ConversionError("source layout evidence changed", opcode="tt.map_elementwise")
    result = {row["id"]: row for row in witness["maps"]}
    if len(result) != len(maps) or set(result) != {op.attributes[KEY].data for op in maps}:
        raise ConversionError("source layout map roster changed", opcode="tt.map_elementwise")
    for row in result.values():
        validate(row)
    witness["source_ir"] = source.read_text()
    validate_witness(witness)
    return result, witness
