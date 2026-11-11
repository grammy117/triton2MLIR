#!/usr/bin/env python3
"""Observe complete source effects and results through Merlin's target backend."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import qualify_spike_model as Q

SCHEMA = "t2m.spike-effect-qualification.v1"


def observe(mode, root, out):
    """Worker: observe actual execution without reading any expected outcome."""
    from merlin.llvmlower.runtime_invocation import resolve
    from merlin.runtime.backends import spike_model
    from merlin.runtime.effect_observation import decode
    from merlin.runtime.input_observation import verify_observed_inputs
    from merlin.runtime.whole_model_readback import decode_outputs

    out = Path(out)
    build = json.loads((out / "build.json").read_text())
    info = build["build"]
    config = info["effect_observation"]
    if config != {"schema": "merlin.source-effect-observation.v1", "max_bytes": 4 << 20,
                  "stdout_frame": "source_stdout", "assertion_exit_code": 134}:
        raise ValueError("unexpected source-effect observation selection")
    if json.loads((out / "image/cgen/effect-observation.json").read_text()) != config:
        raise ValueError("source effect observer differs from its build preparation")
    abi = resolve(out / "image-input/model.mlir", out / "compiler/signature.json", build["runtime_input_path"])
    if info["runtime_invocation_abi"] != abi or json.loads((out / "image/cgen/runtime-abi.json").read_text()) != abi:
        raise ValueError("effect runtime ABI differs from actual source types and invocation storage")
    if mode == "run":
        for item in build["tools"].values():
            if Q.pin(item["path"]) != {k: item[k] for k in ("bytes", "sha256")}:
                raise ValueError("selected tool changed after compilation")
        os.environ.update(MERLIN_SPIKE=build["tools"]["spike"]["path"], MERLIN_NM=build["tools"]["nm"]["path"])
        process = spike_model.run_process(info["elf"], mem_bytes=info["mem_bytes"], vlen=info["vlen"], timeout=30)
        (out / "stdout.bin").write_bytes(process.stdout)
        (out / "simulator-stderr.bin").write_bytes(process.stderr)
        Q.write_json(out / "target-process.json", {"returncode": process.returncode, "timed_out": False,
            "stdout": Q.pin(out / "stdout.bin"), "stderr": Q.pin(out / "simulator-stderr.bin")})
    process = json.loads((out / "target-process.json").read_text())
    if (process.get("timed_out") is not False or process["stdout"] != Q.pin(out / "stdout.bin")
            or process["stderr"] != Q.pin(out / "simulator-stderr.bin")):
        raise ValueError("effect target process is incomplete or its byte streams changed")
    observed = decode((out / "stdout.bin").read_bytes(), process["returncode"], max_bytes=config["max_bytes"])
    values, frames = observed["outputs"], observed["frames"]
    if abi["inputs"]:
        values, frames, readonly = verify_observed_inputs(info["readonly_input_observation"], values, frames)
    else:
        readonly = {"arguments": [], "not_required": "source-has-no-inputs"}
    returning = observed["termination"] == "return"
    specs = [(row["shape"], row["dtype"]) for row in abi["outputs"]]
    arrays = decode_outputs(values, frames, specs if returning else [])
    for i, (_, dtype) in enumerate(specs if returning else []):
        if dtype == "bf16": arrays[i] = (arrays[i].view(np.uint32) >> 16).astype(np.uint16)
        elif dtype == "i1":
            if np.any(arrays[i] > 1): raise ValueError("noncanonical Boolean result storage")
            arrays[i] = arrays[i].astype(np.bool_)
    run = {"execution": "observed", "termination": observed["termination"], "invocations": 1,
        "elf": Q.pin(out / "image/model.elf"), "stdout": Q.pin(out / "stdout.bin"),
        "process": Q.pin(out / "target-process.json"), "readonly_inputs": readonly,
        "out_specs": [[list(s), t] for s, t in specs], "declared_result_count": len(specs),
        "result_count": len(arrays), "outputs_published": returning, "metrics": observed["metrics"],
        "accelerator_execution": "not-performed", "timing_qualification": "not-established"}
    if mode == "check":
        if json.loads((out / "run.json").read_text()) != run:
            raise ValueError("effect run record differs from retained target observations")
        for name in ("stdout", "stderr"):
            if (out / f"source-{name}.bin").read_bytes() != observed[name]:
                raise ValueError("retained source stream differs from its actual target frame")
        if returning:
            with np.load(out / "outputs.npz", allow_pickle=False) as saved:
                if set(saved.files) != {f"out{i}" for i in range(len(arrays))}:
                    raise ValueError("effect output roster differs from complete target frames")
                for i, array in enumerate(arrays):
                    other = saved[f"out{i}"]
                    if (array.shape, array.dtype, array.tobytes()) != (other.shape, other.dtype, other.tobytes()):
                        raise ValueError("effect output differs from its actual target frame")
        elif (out / "outputs.npz").exists():
            raise ValueError("aborted target invocation cannot publish tensor results")
        return
    for name in ("stdout", "stderr"): (out / f"source-{name}.bin").write_bytes(observed[name])
    if returning: np.savez(out / "outputs.npz", **{f"out{i}": value for i, value in enumerate(arrays)})
    Q.write_json(out / "run.json", run)


def reference_context(capture, inputs, reference):
    from t2m.effect_reference import validate_reference, validate_inputs
    from t2m.effect_results import load_archive, validate_results
    from t2m.shared_tensor import TensorABI
    spec = TensorABI.from_dict(json.loads((capture / "spec.json").read_text()))
    expected = validate_reference(json.loads(reference.read_text()), json.loads((capture / "provenance.json").read_text()))
    with np.load(inputs, allow_pickle=False) as archive:
        if set(archive.files) != {f"in{i}" for i in range(len(spec.inputs))}:
            raise ValueError("effect inputs need every original argument exactly once")
        arrays = validate_inputs(spec, [archive[f"in{i}"] for i in range(len(spec.inputs))])
    if expected["termination"] == "return":
        if spec.outputs and expected["schema"] != "t2m.effect-reference.v2":
            raise ValueError("returning target tensors require every numeric reference")
        if expected["schema"] == "t2m.effect-reference.v2":
            validate_results(spec, arrays, load_archive(reference.with_name("reference.npz"), len(spec.outputs)),
                             expected["results"]["comparisons"])
    return spec, expected, arrays


def compare(capture, out, inputs, reference):
    from t2m.effect_results import compare_results
    spec, expected, arrays = reference_context(capture, inputs, reference)
    run = json.loads((out / "run.json").read_text())
    if run["termination"] != expected["termination"] or any(
            (out / f"source-{key}.bin").read_bytes() != expected[key].encode("utf-8") for key in ("stdout", "stderr")):
        raise ValueError("actual target source effects differ from their independent reference")
    count = 0
    if expected["termination"] == "return":
        if spec.outputs:
            count = compare_results(out / "outputs.npz", reference.with_name("reference.npz"), expected, spec, arrays)
        else:
            with np.load(out / "outputs.npz", allow_pickle=False) as archive:
                if archive.files: raise ValueError("void target published unexpected tensors")
    elif (out / "outputs.npz").exists():
        raise ValueError("aborted target published completed tensors")
    return {"reference_comparison": "passed", "termination": expected["termination"], "result_count": count,
        "declared_result_count": len(spec.outputs), "outputs_published": expected["termination"] == "return",
        **({"result_comparisons": expected["results"]["comparisons"]} if "results" in expected else {})}


def qualify(capture, out, *, merlin_root, python=sys.executable, inputs=None, reference=None):
    from t2m.bundle import verify_bundle
    from t2m.effect_results import reference_members
    capture, out, root = Path(capture).resolve(), Path(out).absolute(), Path(merlin_root).resolve()
    receipt = verify_bundle(capture)
    if receipt["schema"] != "t2m.effect-kernel-receipt.v1":
        raise ValueError("source-effect target qualification requires an actual effect package")
    if out.exists() or out.is_symlink() or capture in out.parents:
        raise ValueError("effect qualification output must be new and outside the source")
    if (inputs is None) != (reference is None):
        raise ValueError("changed effect invocation requires inputs and complete references together")
    inputs = Path(inputs).absolute() if inputs is not None else capture / "inputs.npz"
    reference = Path(reference).absolute() if reference is not None else capture / "reference.json"
    _, expected, _ = reference_context(capture, inputs, reference)
    paths = {"inputs": inputs, "reference": reference}
    if "reference.npz" in reference_members(expected): paths["numeric_reference"] = reference.with_name("reference.npz")
    if any(p.is_symlink() or not p.is_file() for p in paths.values()):
        raise ValueError("effect invocation archives must be direct regular files")
    invocation = {name: {"path": str(path.resolve()), **Q.pin(path)} for name, path in paths.items()}
    original = Q.pin(capture / "capture_receipt.json")
    out.mkdir(parents=True, exist_ok=False)
    (out / "input-receipt.json").write_bytes((capture / "capture_receipt.json").read_bytes())
    Q.run_worker("stage", root, out, [capture], python)
    Q.run_worker("build", root, out, [inputs, '["reuse_loop_tensor_destinations"]'], python)
    Q.run_worker("run", root, out, [], python)
    evaluation = compare(capture, out, inputs, reference)
    if verify_bundle(capture) != receipt or Q.pin(capture / "capture_receipt.json") != original:
        raise ValueError("original source effect package changed during target qualification")
    Q.write_json(out / "evaluation.json", evaluation)
    report = {"schema": SCHEMA, "status": "matched-expected-outcome", "execution_scope": "functional-riscv-source-effects",
        "source_directory": str(capture), "source_receipt": original, "invocation": invocation, "evaluation": evaluation,
        "artifacts": {str(p.relative_to(out)): Q.pin(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    Q.write_json(out / "qualification.json", report)
    try: verify(out, merlin_root=root, python=python)
    except BaseException:
        (out / "qualification.json").unlink(missing_ok=True)
        raise
    return report


def verify(out, *, merlin_root, python=sys.executable):
    from t2m.bundle import verify_bundle
    out = Path(out)
    report = json.loads((out / "qualification.json").read_text())
    if (report.get("schema") != SCHEMA or report.get("status") != "matched-expected-outcome"
            or report.get("execution_scope") != "functional-riscv-source-effects"):
        raise ValueError("not a completed source-effect target qualification")
    required = {"input-receipt.json", "build.json", "run.json", "evaluation.json", "target-process.json", "stdout.bin",
        "simulator-stderr.bin", "source-stdout.bin", "source-stderr.bin", "image/model.elf",
        "image/cgen/runtime-abi.json", "image/cgen/effect-observation.json", "compiler/compile-inputs.json"}
    if not required <= set(report["artifacts"]): raise ValueError("incomplete effect target artifact roster")
    for name, value in report["artifacts"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or name == "qualification.json":
            raise ValueError("invalid effect target evidence path")
        if any(out.joinpath(*relative.parts[:i]).is_symlink() for i in range(len(relative.parts) + 1)) or Q.pin(out / name) != value:
            raise ValueError("effect target evidence changed: " + name)
    capture = Path(report["source_directory"])
    source = verify_bundle(capture)
    if (source["schema"] != "t2m.effect-kernel-receipt.v1" or Q.pin(capture / "capture_receipt.json") != report["source_receipt"]
            or Q.pin(out / "input-receipt.json") != report["source_receipt"]):
        raise ValueError("effect target does not bind its unchanged original source package")
    stage = Q.checked_stage(out / "compiler")
    Q.checked_stage(out / "image-input", model_alias=True)
    if (stage["source_profile"] != "standalone-effect-kernel"
            or stage["source"]["capture_receipt_sha256"] != report["source_receipt"]["sha256"]
            or Q.pin(out / "compiler/compile-inputs.json") != Q.pin(out / "image-input/compile-inputs.json")
            or Q.pin(out / "compiler/program.mlir") != Q.pin(out / "image-input/model.mlir")):
        raise ValueError("effect compiler projection differs from its source package")
    invocation = report["invocation"]
    if set(invocation) not in ({"inputs", "reference"}, {"inputs", "reference", "numeric_reference"}):
        raise ValueError("effect target needs the complete invocation/reference binding")
    for record in invocation.values():
        path = Path(record["path"])
        if not path.is_absolute() or path.is_symlink() or Q.pin(path) != {k: record[k] for k in ("bytes", "sha256")}:
            raise ValueError("effect invocation changed after qualification")
    inputs, reference = (Path(invocation[name]["path"]) for name in ("inputs", "reference"))
    _, expected, _ = reference_context(capture, inputs, reference)
    if ("results" in expected) != ("numeric_reference" in invocation) or "numeric_reference" in invocation and (
            Path(invocation["numeric_reference"]["path"]) != reference.with_name("reference.npz")):
        raise ValueError("effect target numeric references do not match their complete reference contract")
    build = json.loads((out / "build.json").read_text())
    run = json.loads((out / "run.json").read_text())
    signature = json.loads((out / "compiler/signature.json").read_text())
    if (build["runtime_input_path"] != str(inputs.resolve()) or build["runtime_inputs"] != Q.pin(inputs)
            or build["source_program"] != Q.pin(out / "compiler/program.mlir")
            or build.get("reference_values_supplied_to_build") is not False
            or [row["original_index"] for row in run["readonly_inputs"]["arguments"]] != list(range(len(signature["inputs"])))
            or not all(row.get("passed") is True for row in run["readonly_inputs"]["arguments"])):
        raise ValueError("effect target compiler consumed different source or invocation inputs")
    for mode in ("stage", "build", "run"):
        if json.loads((out / f"{mode}-process.json").read_text())["returncode"] != 0:
            raise ValueError("effect qualification contains an unsuccessful observer process")
    command = [str(python), str(Path(Q.__file__).resolve()), "--worker", "check", str(Path(merlin_root).resolve()), str(out.resolve())]
    checked = subprocess.run(command, capture_output=True, timeout=30, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    if checked.returncode: raise ValueError("retained effect wire failed: " + checked.stderr[-3000:].decode(errors="replace"))
    evaluation = compare(capture, out, inputs, reference)
    if evaluation != report["evaluation"] or json.loads((out / "evaluation.json").read_text()) != evaluation:
        raise ValueError("retained effect evaluation differs from its complete independent references")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "out", "merlin-root", "python", "inputs", "reference"):
        parser.add_argument("--" + name, type=Path, required=name in ("out", "merlin-root"),
                            default=Path(sys.executable) if name == "python" else None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify: report = verify(args.out, merlin_root=args.merlin_root, python=args.python)
    else:
        if args.capture is None: parser.error("--capture is required for execution")
        report = qualify(args.capture, args.out, merlin_root=args.merlin_root, python=args.python,
                         inputs=args.inputs, reference=args.reference)
    print(json.dumps({key: report[key] for key in ("schema", "status", "evaluation")}))


if __name__ == "__main__": main()
