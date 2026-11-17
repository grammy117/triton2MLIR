"""Explicit staged execution-result transport in compiler/evaluator workers.

Compiler entrypoints receive only compiler-owned members. Runtime arrays and
executed descriptors stay in the evaluator; no output-size prediction occurs.
"""

import hashlib
import json
import platform
from pathlib import Path


def _record(path):
    raw = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def compile(staged, work, source, module, signature, features):
    from merlin.frontends.execution_result_signature import PROFILE, validate_signature
    from merlin.llvmlower.owned_result_abi import pipeline
    from merlin.llvmlower.owned_result_program import save
    from merlin.llvmlower.owned_results import compile_owned_results
    from merlin.llvmlower.toolchain import clang, compiler_python

    validate_signature(module, signature)
    manifest = json.loads((staged / "compile-inputs.json").read_text())
    if manifest.get("result_mode") != PROFILE:
        raise ValueError(
            "execution-produced results require their explicit compiler selection"
        )
    policy = signature["allocation_policy"]
    program = compile_owned_results(
        source,
        workdir=work,
        entry=signature["entry"],
        features=features,
        timeout=30,
        max_bytes=policy["max_bytes"],
        max_allocations=policy["max_allocations"],
        max_output_bytes=policy["max_output_bytes"],
        allow_existing_workdir=True,
    )
    saved = save(program)
    report = {
        "schema": "t2m.merlin-host-compilation.v2",
        "result_abi": PROFILE,
        "compiler_input_staging": "passed",
        "whole_body_lowering": "passed",
        "native_object_and_link": "passed",
        "source_program": _record(staged / "program.mlir"),
        "source_signature": _record(staged / "signature.json"),
        "prepared_program": _record(work / "prepared.mlir"),
        "llvm": _record(work / "model.ll"),
        "shared_library": _record(work / "model.so"),
        "owned_result_program": _record(saved),
        "pipeline": pipeline(features=features),
        "features": sorted(features),
        "lowering_api": "merlin.llvmlower.owned_results.compile_owned_results",
        "codegen_api": "merlin.llvmlower.codegen.build_host_shared",
        "compiler_python": str(compiler_python()),
        "codegen_tool": str(clang()),
        "host_machine": platform.machine(),
        "evaluator_data_supplied_to_compiler": False,
        "accelerator_execution": "not-performed",
    }
    (work / "compilation.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )


def execute(
    staged, work, arguments, arrays, signature, compiled, evaluation, *, observed=False
):
    import numpy as np
    from merlin.frontends.execution_result_signature import PROFILE
    from merlin.llvmlower.execution_result_inputs import resolve
    from merlin.llvmlower.owned_result_program import FILENAME, load

    if (
        compiled.get("schema") != "t2m.merlin-host-compilation.v2"
        or compiled.get("result_abi") != PROFILE
        or compiled.get("owned_result_program") != _record(work / FILENAME)
        or compiled.get("shared_library") != _record(work / "model.so")
    ):
        raise ValueError(
            "execution result library/program differs from its compiler selection"
        )
    manifest = json.loads((staged / "compile-inputs.json").read_text())
    if manifest.get("result_mode") != PROFILE:
        raise ValueError(
            "execution result invocation lacks its explicit compiler result selection"
        )
    source = (staged / "program.mlir").read_text()
    program = load(work / FILENAME)
    if program.source.read_text() != source or program.entry != signature["entry"]:
        raise ValueError(
            "executed result program differs from the original staged source"
        )
    policy = signature["allocation_policy"]
    if any(
        getattr(program, name) != policy[name]
        for name in ("max_bytes", "max_allocations", "max_output_bytes")
    ):
        raise ValueError(
            "execution result allocation policy differs from its compiled selection"
        )
    invocation = resolve(source, signature, arrays)
    if observed:
        from process_outcome import started

        started(
            evaluation,
            backend="merlin-host",
            program=staged / "program.mlir",
            arguments=arguments,
            implementation={
                "kind": "shared-library",
                "artifact": _record(work / "model.so"),
            },
        )
    before = [a.tobytes() for a in arrays]
    previous = None
    reports = []
    for _ in range(1 if observed else 2):
        outputs, report = program.execute(invocation)
        if any(a.tobytes() != old for a, old in zip(arrays, before, strict=True)):
            raise ValueError(
                "executed result function mutated a functional tensor input"
            )
        actual = [(a.shape, a.dtype.str, a.tobytes()) for a in outputs]
        if previous is not None and actual != previous:
            raise ValueError(
                "repeated execution changed result shapes or complete payloads"
            )
        previous = actual
        reports.append(report)
    np.savez(
        evaluation / "outputs.npz", **{f"out{i}": a for i, a in enumerate(outputs)}
    )
    (evaluation / "evaluation.json").write_text(
        json.dumps(
            {
                "schema": "t2m.merlin-host-evaluation.v2",
                "result_abi": PROFILE,
                "invocations": len(reports),
                "functional_inputs_preserved": True,
                "result_observations": reports,
                "outputs": _record(evaluation / "outputs.npz"),
                "arguments": _record(arguments),
                "source_program": _record(staged / "program.mlir"),
                "source_signature": _record(staged / "signature.json"),
                "shared_library": _record(work / "model.so"),
            },
            indent=2,
        )
        + "\n"
    )
