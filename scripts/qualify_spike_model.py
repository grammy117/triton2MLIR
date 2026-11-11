#!/usr/bin/env python3
"""Qualify a bound model through Merlin's ordinary whole-model Spike backend.

This explicit execution edge supplies no target cases to TTIR conversion.
Run with the frontend Python; --python selects the existing Merlin environment.
Compiler/GCC/Spike/nm selection uses Merlin's ordinary environment variables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import numpy as np


def pin(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def checked_stage(directory, *, model_alias=False):
    directory = Path(directory)
    manifest = json.loads((directory / "compile-inputs.json").read_text())
    if manifest.get("schema") != "merlin.compile-inputs.v1":
        raise ValueError("expected Merlin compiler-only staging")
    for name, digest in manifest["members"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("invalid compiler member path")
        if any(directory.joinpath(*relative.parts[:i]).is_symlink() for i in range(len(relative.parts) + 1)):
            raise ValueError("compiler member must not use symlinks")
        if pin(directory / name)["sha256"] != digest:
            raise ValueError("compiler member changed: " + name)
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    aliases = {"model.mlir"} if model_alias else set()
    if files != set(manifest["members"]) | {"compile-inputs.json"} | aliases:
        raise ValueError("compiler directory contains undeclared members")
    if model_alias and pin(directory / "model.mlir") != pin(directory / "program.mlir"):
        raise ValueError("runtime filename projection differs from staged program")
    if set(manifest["members"]) & set(manifest.get("excluded_evaluator_members", [])):
        raise ValueError("compiler staging contains evaluator data")
    return manifest


def worker(mode, root, out, *arguments):
    sys.path.insert(0, str(Path(root) / "src"))
    out = Path(out)
    if mode == "stage":
        from merlin.frontends.compile_inputs import stage_compile_inputs
        stage_compile_inputs(arguments[0], out / "compiler")
        checked_stage(out / "compiler")
    elif mode == "build":
        from merlin.llvmlower import toolchain
        from merlin.runtime.backends import spike, spike_model
        manifest = checked_stage(out / "compiler")
        signature = json.loads((out / "compiler/signature.json").read_text())
        entries = signature["inputs"] + signature["outputs"]
        for entry in entries:
            shape = entry.get("shape")
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError("whole-model image requires static ranked tensor arguments/results")
        if sum(math.prod(e["shape"]) * 8 for e in entries) > 64 << 20:
            raise ValueError("whole-model qualification exceeds the 64 MiB value budget")
        if not signature["inputs"] or not signature["outputs"]:
            raise ValueError("whole-model qualification needs tensor arguments and results")
        source = (out / "compiler/program.mlir").read_bytes()
        projection = out / "image-input"
        shutil.copytree(out / "compiler", projection)
        (projection / "model.mlir").write_bytes(source)
        checked_stage(projection, model_alias=True)
        tools = {name: {"path": str(path.resolve(strict=True)), **pin(path)} for name, path in (
            ("clang", toolchain.clang()), ("compiler_python", toolchain.compiler_python()),
            ("gcc", spike.gcc_path()), ("spike", spike.spike_path()), ("nm", toolchain.nm()))}
        report = spike_model.build(projection, out / "image", inputs_npz=arguments[0],
            backend="scalar", features=frozenset(json.loads(arguments[1])),
            full_readback=True, readonly_input_indices=tuple(range(len(signature["inputs"]))),
            arena_mb=4, dram_bytes=64 << 20, code_reserve=2 << 20)
        checked_stage(out / "compiler")
        if (projection / "model.mlir").read_bytes() != source:
            raise ValueError("runtime filename projection changed the staged program")
        write_json(out / "build.json", {"build": report, "tools": tools,
            "driver": pin(__file__), "selected_features": json.loads(arguments[1]),
            "merlin_sources": {name: pin(Path(root) / name) for name in (
                "src/merlin/runtime/backends/spike_model.py", "src/merlin/runtime/backends/_model_elf_metadata.py",
                "src/merlin/runtime/whole_model_readback.py", "src/merlin/runtime/out_bin.py",
                "src/merlin/runtime/input_observation.py", "src/merlin/llvmlower/c_runtime.py",
                "src/merlin/llvmlower/pipeline.py", "src/merlin/llvmlower/toolchain.py",
                "src/merlin/frontends/linalg_mlir.py", "src/merlin/frontends/linalg_named_regions.py",
                "src/merlin/xdsl_dialects/standard_linalg.py")},
            "dependency_source_closure": "not-established",
            "source_program": pin(out / "compiler/program.mlir"),
            "runtime_inputs": pin(arguments[0]), "runtime_inputs_embedded_in_elf": True,
            "runtime_input_consumer": "merlin.llvmlower.c_runtime.generate",
            "compiler_directory_contains_evaluator_data": False,
            "reference_values_supplied_to_build": False,
            "compiler_staging_source": manifest["source"]})
    elif mode in ("run", "check"):
        from merlin.llvmlower.c_runtime import _out_specs
        from merlin.runtime.backends import spike_model
        from merlin.runtime.input_observation import verify_observed_inputs
        from merlin.runtime.out_bin import parse_binary_console_details
        from merlin.runtime.whole_model_readback import decode_outputs
        build = json.loads((out / "build.json").read_text())
        tools = build["tools"]
        report = build["build"]
        if mode == "run":
            for item in tools.values():
                if pin(item["path"]) != {k: item[k] for k in ("bytes", "sha256")}:
                    raise ValueError("selected tool changed after compilation")
            # Select the exact simulator and metadata reader used by the image build.
            os.environ.update(MERLIN_SPIKE=tools["spike"]["path"], MERLIN_NM=tools["nm"]["path"])
            raw = spike_model.run_raw(report["elf"], mem_bytes=report["mem_bytes"], vlen=report["vlen"], timeout=30)
            (out / "stdout.bin").write_bytes(raw)
        else:
            raw = (out / "stdout.bin").read_bytes()
        outputs, metrics, frames = parse_binary_console_details(raw)
        outputs, frames, observed = verify_observed_inputs(report["readonly_input_observation"], outputs, frames)
        specs = _out_specs(out / "image-input/model.mlir")
        arrays = decode_outputs(outputs, frames, specs)
        for i, (_, dtype) in enumerate(specs):
            if dtype == "bf16":
                arrays[i] = (arrays[i].view(np.uint32) >> 16).astype(np.uint16)
            elif dtype == "i1":
                if np.any(arrays[i] > 1):
                    raise ValueError("Boolean result contains a non-Boolean source word")
                arrays[i] = arrays[i].astype(np.bool_)
        if mode == "check":
            with np.load(out / "outputs.npz", allow_pickle=False) as stored:
                if set(stored.files) != {f"out{i}" for i in range(len(arrays))}:
                    raise ValueError("retained output roster differs from raw frames")
                for i, value in enumerate(arrays):
                    saved = stored[f"out{i}"]
                    if (value.shape, value.dtype, value.tobytes()) != (saved.shape, saved.dtype, saved.tobytes()):
                        raise ValueError("retained output values differ from raw frames")
            run = json.loads((out / "run.json").read_text())
            if run["readonly_inputs"] != observed or run["metrics"] != metrics or run["out_specs"] != [[list(s), t] for s, t in specs]:
                raise ValueError("retained run observations differ from raw frames")
            return
        np.savez(out / "outputs.npz", **{f"out{i}": value for i, value in enumerate(arrays)})
        write_json(out / "run.json", {"execution": "passed", "elf": pin(report["elf"]),
            "stdout": pin(out / "stdout.bin"), "outputs": pin(out / "outputs.npz"),
            "metrics": metrics, "readonly_inputs": observed, "out_specs": specs,
            "isa": "rv64gcv_zfh_zvfh", "harts": 1, "timeout_seconds": 30,
            "timing_qualification": "not-established", "accelerator_execution": "not-performed"})
    else:
        raise ValueError("unknown qualification worker")


def run_worker(mode, root, out, arguments, python):
    command = [str(python), str(Path(__file__).resolve()), "--worker", mode, str(root), str(out), *map(str, arguments)]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MERLIN_COMPILE_TIMEOUT_S="30", TMPDIR=str(out))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        (out / f"{mode}.stdout").write_bytes(stdout)
        (out / f"{mode}.stderr").write_bytes(stderr)
        raise ValueError("whole-model " + mode + " exceeded 60 seconds") from exc
    (out / f"{mode}.stdout").write_bytes(stdout)
    (out / f"{mode}.stderr").write_bytes(stderr)
    write_json(out / f"{mode}-process.json", {"argv": command, "returncode": process.returncode})
    if process.returncode:
        raise ValueError("whole-model " + mode + " failed: " + stderr[-3000:].decode(errors="replace"))


def references(capture):
    from xdsl.dialects import func
    from t2m.model_bundle import _references
    from t2m.shared_tensor import parse_shared
    module = parse_shared((capture / "model.mlir").read_text())
    if any(op.name == "vector.print" for op in module.walk()):
        raise ValueError("numeric model qualification requires a separate reference for source print effects")
    entries = [op for op in module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"]
    if len(entries) != 1:
        raise ValueError("expected exactly one model entry")
    parent = json.loads((capture / "parent/capture_receipt.json").read_text())
    return _references(capture, parent, entries[0].function_type.outputs, metadata=capture / "parent/meta.json")


def compare(capture, out, rtol, atol):
    from t2m.tensor_storage import assert_tensor_close
    if not all(np.isfinite(x) and x >= 0 for x in (rtol, atol)):
        raise ValueError("comparison tolerances must be finite and nonnegative")
    expected, reference_members = references(capture)
    signature = json.loads((out / "compiler/signature.json").read_text())
    with np.load(out / "outputs.npz", allow_pickle=False) as outputs:
        if set(outputs.files) != {f"out{i}" for i in range(len(expected))}:
            raise ValueError("target output archive does not contain every model result")
        if len(signature["outputs"]) != len(expected):
            raise ValueError("staged signature differs from complete reference tuple")
        for i, (golden, spec) in enumerate(zip(expected, signature["outputs"], strict=True)):
            actual = outputs[f"out{i}"]
            if (actual.shape, actual.dtype) != (golden.shape, golden.dtype):
                raise ValueError(f"target result {i} differs from reference shape/dtype")
            assert_tensor_close(actual, golden, spec["dtype"], rtol=rtol, atol=atol)
    return {"reference_comparison": "passed", "result_count": len(expected),
        "reference_members": reference_members, "rtol": rtol, "atol": atol}


def qualify(capture, out, *, merlin_root, python=sys.executable, rtol=0, atol=0, features=()):
    from t2m.bundle import verify_bundle
    capture, out, root = Path(capture).resolve(), Path(out).absolute(), Path(merlin_root).resolve()
    if out.exists() or out.is_symlink() or capture in out.parents:
        raise ValueError("qualification output must be new and outside the source bundle")
    if not all(np.isfinite(x) and x >= 0 for x in (rtol, atol)):
        raise ValueError("comparison tolerances must be finite and nonnegative")
    receipt = verify_bundle(capture)
    if receipt["schema"] != "t2m.bound-model-receipt.v1":
        raise ValueError("this whole-model execution edge requires a bound model package")
    references(capture)  # Check complete source references/effect scope before building.
    original = pin(capture / "capture_receipt.json")
    out.mkdir(parents=True, exist_ok=False)
    (out / "input-receipt.json").write_bytes((capture / "capture_receipt.json").read_bytes())
    run_worker("stage", root, out, [capture], python)
    run_worker("build", root, out, [capture / "inputs.npz", json.dumps(sorted(set(features) | {"reuse_loop_tensor_destinations"}))], python)
    run_worker("run", root, out, [], python)
    evaluation = compare(capture, out, rtol, atol)
    if verify_bundle(capture) != receipt or pin(capture / "capture_receipt.json") != original:
        raise ValueError("source model changed during target qualification")
    write_json(out / "evaluation.json", evaluation)
    report = {"schema": "t2m.spike-model-qualification.v1", "source_directory": str(capture),
        "source_receipt": original, "execution_scope": "functional-riscv-whole-model",
        "status": "passed", "evaluation": evaluation,
        "artifacts": {str(p.relative_to(out)): pin(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    write_json(out / "qualification.json", report)
    try:
        verify(out, merlin_root=root, python=python)
    except BaseException:
        (out / "qualification.json").unlink(missing_ok=True)
        raise
    return report


def verify(out, *, merlin_root, python=sys.executable):
    from t2m.bundle import verify_bundle
    out = Path(out)
    report = json.loads((out / "qualification.json").read_text())
    if (report.get("schema") != "t2m.spike-model-qualification.v1" or report.get("status") != "passed"
            or report.get("execution_scope") != "functional-riscv-whole-model"):
        raise ValueError("not a completed whole-model target qualification")
    required = {"input-receipt.json", "compiler/compile-inputs.json", "compiler/program.mlir", "compiler/signature.json",
        "image-input/model.mlir", "image/model.elf", "build.json", "run.json", "stdout.bin", "outputs.npz", "evaluation.json"}
    if not required <= set(report["artifacts"]):
        raise ValueError("incomplete target qualification artifact roster")
    for name, value in report["artifacts"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or name == "qualification.json":
            raise ValueError("invalid target evidence path")
        if any(out.joinpath(*relative.parts[:i]).is_symlink() for i in range(len(relative.parts) + 1)) or pin(out / name) != value:
            raise ValueError("target evidence changed: " + name)
    capture = Path(report["source_directory"])
    verify_bundle(capture)
    if pin(capture / "capture_receipt.json") != report["source_receipt"] or pin(out / "input-receipt.json") != report["source_receipt"]:
        raise ValueError("target evidence does not bind the unchanged source receipt")
    stage = checked_stage(out / "compiler")
    if stage["source"]["capture_receipt_sha256"] != report["source_receipt"]["sha256"]:
        raise ValueError("compiler stage does not bind source receipt")
    if pin(out / "compiler/program.mlir") != pin(out / "image-input/model.mlir"):
        raise ValueError("runtime projection differs from staged program")
    if pin(out / "compiler/compile-inputs.json") != pin(out / "image-input/compile-inputs.json"):
        raise ValueError("runtime projection changed compiler member bindings")
    checked_stage(out / "image-input", model_alias=True)
    build = json.loads((out / "build.json").read_text())
    run = json.loads((out / "run.json").read_text())
    signature = json.loads((out / "compiler/signature.json").read_text())
    if (run.get("execution") != "passed" or run["elf"] != pin(out / "image/model.elf")
            or run["stdout"] != pin(out / "stdout.bin") or run["outputs"] != pin(out / "outputs.npz")
            or build["source_program"] != pin(out / "compiler/program.mlir")
            or build["runtime_inputs"] != pin(capture / "inputs.npz")
            or run.get("accelerator_execution") != "not-performed"
            or run.get("timing_qualification") != "not-established"
            or [row["original_index"] for row in run["readonly_inputs"]["arguments"]] != list(range(len(signature["inputs"])))
            or not all(row.get("passed") is True for row in run["readonly_inputs"]["arguments"])
            or build.get("reference_values_supplied_to_build") is not False):
        raise ValueError("inconsistent target source/image/execution evidence")
    for mode in ("stage", "build", "run"):
        if json.loads((out / f"{mode}-process.json").read_text())["returncode"] != 0:
            raise ValueError("target qualification contains a failed process")
    # Re-decode retained wire bytes with an explicitly selected existing Merlin.
    # Verification neither runs the simulator nor executes tools named by JSON.
    command = [str(python), str(Path(__file__).resolve()), "--worker", "check", str(Path(merlin_root).resolve()), str(out.resolve())]
    process = subprocess.run(command, capture_output=True, timeout=30,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    if process.returncode:
        raise ValueError("retained target wire evidence failed: " + process.stderr[-3000:].decode(errors="replace"))
    evaluation = report["evaluation"]
    if json.loads((out / "evaluation.json").read_text()) != evaluation:
        raise ValueError("inconsistent target evaluation record")
    if compare(capture, out, evaluation["rtol"], evaluation["atol"]) != evaluation:
        raise ValueError("target comparison evidence differs from complete result references")
    return report


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merlin-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--rtol", type=float, default=0)
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if args.merlin_root is None:
            parser.error("--merlin-root is required to verify retained wire frames")
        report = verify(args.out, merlin_root=args.merlin_root, python=args.python)
    else:
        if args.capture is None or args.merlin_root is None:
            parser.error("--capture and --merlin-root are required for execution")
        report = qualify(args.capture, args.out, merlin_root=args.merlin_root, python=args.python,
            rtol=args.rtol, atol=args.atol, features=args.feature)
    print(json.dumps({"schema": report["schema"], "status": report["status"], "evaluation": report["evaluation"]}))


if __name__ == "__main__":
    main()
