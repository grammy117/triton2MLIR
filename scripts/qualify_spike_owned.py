"""Explicit signature-v6 target orchestration; references never enter builds."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import qualify_spike_model as Q

SCHEMA = "t2m.spike-owned-result-qualification.v1"
SOURCES = (
    "src/merlin/runtime/backends/owned_result_model.py", "src/merlin/runtime/backends/spike_model.py",
    "src/merlin/runtime/backends/host_compiler_policy.py", "src/merlin/runtime/backends/_model_elf_metadata.py",
    "src/merlin/runtime/owned_result_readback.py", "src/merlin/runtime/out_bin.py",
    "src/merlin/llvmlower/owned_result_driver.py", "src/merlin/llvmlower/owned_result_lowering.py",
    "src/merlin/llvmlower/owned_result_abi.py", "src/merlin/llvmlower/owned_result_c.py",
    "src/merlin/llvmlower/owned_results.py", "src/merlin/llvmlower/owned_result_allocations.py",
    "src/merlin/llvmlower/dealloc_lifetime.py", "src/merlin/llvmlower/dealloc_promotions.py",
    "src/merlin/llvmlower/runtime_invocation.py", "src/merlin/llvmlower/compilation_recipe.py",
    "src/merlin/llvmlower/execution_result_inputs.py", "src/merlin/frontends/execution_result_signature.py",
    "src/merlin/frontends/execution_call_views.py", "src/merlin/frontends/execution_call_contracts.py",
    "src/merlin/frontends/execution_model_correspondence.py", "src/merlin/frontends/derived_source_views.py",
    "src/merlin/frontends/execution_runtime_views.py",
    "src/merlin/frontends/execution_source_coordinates.py",
    "src/merlin/frontends/execution_functional_state.py",
    "src/merlin/frontends/execution_storage_views.py",
    "merlin/runtime/c/owned_result_driver.c", "merlin/runtime/c/owned_result_driver.h",
    "merlin/runtime/c/owned_result_readback.c", "merlin/runtime/c/owned_result_readback.h",
    "merlin/runtime/abi/owned_results.c", "merlin/runtime/abi/owned_result_views.c",
    "merlin/runtime/abi/owned_result_views.h", "merlin/runtime/abi/mlir_runtime.c", "merlin/runtime/abi/half_extend.h",
    "merlin/runtime/baremetal/spike/owned_result_main.c", "merlin/runtime/baremetal/spike/crt.S",
    "merlin/runtime/baremetal/spike/htif.c", "merlin/runtime/baremetal/spike/htif.h",
    "merlin/runtime/baremetal/spike/libc_min.c", "merlin/runtime/baremetal/spike/merlin_malloc.c",
    "merlin/runtime/baremetal/spike/model_link.ld",
)


def build(root, out, inputs, features):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike, spike_model

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "t2m"))
    from consumer_features import select_features
    manifest = Q.checked_stage(out / "compiler")
    source = (out / "compiler/program.mlir").read_bytes()
    module = parse_mlir_text(source.decode())
    module.verify()
    requested = json.loads(features)
    selected = select_features(module) | frozenset(requested)
    projection = out / "image-input"
    shutil.copytree(out / "compiler", projection)
    (projection / "model.mlir").write_bytes(source)
    Q.checked_stage(projection, model_alias=True)
    tools = {name: {"path": str(path.resolve(strict=True)), **Q.pin(path)} for name, path in (
        ("clang", toolchain.clang()), ("compiler_python", toolchain.compiler_python()),
        ("gcc", spike.gcc_path()), ("spike", spike.spike_path()), ("nm", toolchain.nm()))}
    signature = json.loads((out / "compiler/signature.json").read_text())
    report = spike_model.build(projection, out / "image", inputs_npz=inputs,
        runtime_signature=out / "compiler/signature.json", backend="scalar", features=selected,
        full_readback=True, readonly_input_indices=tuple(range(len(signature["inputs"]))),
        max_readonly_input_bytes=64 << 20, arena_mb=4, dram_bytes=64 << 20, code_reserve=2 << 20)
    Q.checked_stage(out / "compiler")
    Q.checked_stage(projection, model_alias=True)
    Q.write_json(out / "build.json", {"build": report, "tools": tools,
        "driver": Q.pin(Path(__file__).with_name("qualify_spike_model.py")), "owned_driver": Q.pin(__file__),
        "selected_features": sorted(selected), "requested_features": requested,
        "feature_selector": Q.pin(Path(__file__).resolve().parents[1] / "t2m/consumer_features.py"),
        "merlin_sources": {name: Q.pin(Path(root) / name) for name in SOURCES},
        "dependency_source_closure": "not-established", "source_program": Q.pin(out / "compiler/program.mlir"),
        "runtime_input_path": str(Path(inputs).resolve()), "runtime_inputs": Q.pin(inputs),
        "runtime_inputs_embedded_in_elf": True, "runtime_input_consumer": "merlin.llvmlower.owned_result_driver",
        "compiler_directory_contains_evaluator_data": False, "reference_values_supplied_to_build": False,
        "compiler_staging_source": manifest["source"]})


def observe(mode, root, out):
    from merlin.frontends.compile_inputs import stage_compile_inputs
    from merlin.llvmlower.owned_result_driver import load_arguments
    from merlin.llvmlower.owned_result_driver import verify as verify_driver
    from merlin.runtime.backends import spike_model
    from merlin.runtime.owned_result_readback import PROFILE, decode

    build = json.loads((out / "build.json").read_text())
    report, tools = build["build"], build["tools"]
    source = (out / "compiler/program.mlir").read_text()
    if mode == "check":
        capture = Path(json.loads((out / "qualification.json").read_text())["source_directory"])
        with tempfile.TemporaryDirectory(prefix=".owned-source-replay-", dir=out.parent) as temporary:
            replay = Path(temporary) / "compiler"
            stage_compile_inputs(capture, replay, result_mode="execution-owned")
            expected = Q.checked_stage(replay)
            if expected != Q.checked_stage(out / "compiler"):
                raise ValueError("owned-result compiler stage differs from independent source reconstruction")
            if any((replay / name).read_bytes() != (out / "compiler" / name).read_bytes()
                   for name in expected["members"]):
                raise ValueError("owned-result staged members differ from original source reconstruction")
    signature = json.loads((out / "compiler/signature.json").read_text())
    arrays = load_arguments(source, signature, build["runtime_input_path"])
    expected_driver = verify_driver(source, signature, arrays, out / "image/cgen")
    if report["readback_profile"] != PROFILE or expected_driver != report["driver"]:
        raise ValueError("target driver/readback differs from original source and invocation")
    if json.loads((out / "image/owned-result-image.json").read_text()) != report:
        raise ValueError("target image report differs from its actual build")
    if mode == "run":
        for item in tools.values():
            if Q.pin(item["path"]) != {key: item[key] for key in ("bytes", "sha256")}:
                raise ValueError("selected owned-result tool changed after compilation")
        os.environ.update(MERLIN_SPIKE=tools["spike"]["path"], MERLIN_NM=tools["nm"]["path"])
        isa = spike_model.declared_isa(report["elf"])
        if isa is None or spike_model.declared_harts(report["elf"]) != 1:
            raise ValueError("owned-result image lacks an observed ISA or serial hart declaration")
        process = spike_model.run_process(report["elf"], isa=isa, harts=1,
            mem_bytes=report["mem_bytes"], vlen=report["vlen"], timeout=30)
        (out / "stdout.bin").write_bytes(process.stdout)
        (out / "stderr.bin").write_bytes(process.stderr)
        Q.write_json(out / "target-process.json", {"returncode": process.returncode,
            "argv": process.args, "isa": isa, "harts": 1, "memory": spike_model.declared_memory(report["elf"]),
            "stdout": Q.pin(out / "stdout.bin"), "stderr": Q.pin(out / "stderr.bin"),
            "simulator": tools["spike"], "elf": Q.pin(report["elf"]), "timeout_seconds": 30})
        if process.returncode:
            raise ValueError("owned-result image did not return successfully: " + str(process.returncode))
    process = json.loads((out / "target-process.json").read_text())
    if (process["returncode"] != 0 or process["elf"] != Q.pin(report["elf"])
            or process["stdout"] != Q.pin(out / "stdout.bin") or process["stderr"] != Q.pin(out / "stderr.bin")
            or process["simulator"] != tools["spike"] or process["harts"] != 1):
        raise ValueError("owned-result target process evidence is incomplete or inconsistent")
    if (process["isa"] != spike_model.declared_isa(report["elf"])
            or process["memory"] != list(spike_model.declared_memory(report["elf"]))
            or process["argv"][0] != tools["spike"]["path"] or process["argv"][-1] != report["elf"]):
        raise ValueError("owned-result process metadata differs from its actual image or selected simulator")
    outputs, observed = decode(source, signature, arrays, (out / "stdout.bin").read_bytes())
    run = {"schema": "t2m.execution-owned-target-run.v1", "execution": "passed", "readback": observed,
           "elf": Q.pin(report["elf"]), "stdout": Q.pin(out / "stdout.bin"),
           "target_process": Q.pin(out / "target-process.json"), "isa": process["isa"], "harts": 1,
           "accelerator_execution": "not-performed", "timing_qualification": "not-established"}
    if mode == "run":
        np.savez(out / "outputs.npz", **{f"out{i}": value for i, value in enumerate(outputs)})
        run["outputs"] = Q.pin(out / "outputs.npz")
        Q.write_json(out / "run.json", run)
    else:
        with np.load(out / "outputs.npz", allow_pickle=False) as saved:
            if set(saved.files) != {f"out{i}" for i in range(len(outputs))}:
                raise ValueError("owned-result retained outputs omit an original result")
            for i, value in enumerate(outputs):
                actual = saved[f"out{i}"]
                if (actual.shape, actual.dtype, actual.tobytes()) != (value.shape, value.dtype, value.tobytes()):
                    raise ValueError("owned-result retained outputs differ from actual observed words")
        run["outputs"] = Q.pin(out / "outputs.npz")
        if json.loads((out / "run.json").read_text()) != run:
            raise ValueError("owned-result retained run differs from complete raw observation")


def references(capture, override=None):
    from xdsl.dialects import func

    from t2m.model_bundle import _references
    from t2m.model_input_abi import _npz_shapes
    from t2m.shared_tensor import parse_shared

    receipt = json.loads((capture / "capture_receipt.json").read_text())
    if receipt["schema"] == "t2m.capture-receipt.v2":
        return Q.references(capture, override)
    module = parse_shared((capture / "model.mlir").read_text())
    if any(op.name == "vector.print" for op in module.walk()):
        raise ValueError("owned-result numeric qualification needs a separate source-effect oracle")
    entry = next(op for op in module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward")
    if override is not None:
        headers = _npz_shapes(override)
        if set(headers) != {f"out{i}" for i in range(len(entry.function_type.outputs))}:
            raise ValueError("changed owned-result invocation needs every complete result reference")
        with zipfile.ZipFile(override) as archive:
            if sum(member.file_size for member in archive.infolist()) > (64 << 20) + len(headers) * 16384:
                raise ValueError("owned-result references exceed their complete byte budget")
        with np.load(override, allow_pickle=False) as saved:
            values = [saved[f"out{i}"].copy() for i in range(len(headers))]
        return values, {"invocation-reference.npz": Q.pin(override)}
    # These are independently captured reference dimensions, used only by the
    # evaluator. They are never supplied to source staging, lowering or C generation.
    if len(entry.function_type.outputs) > 1:
        headers = _npz_shapes(capture / "goldens.npz")
        shapes = [headers[f"out{i}"] for i in range(len(entry.function_type.outputs))]
    else:
        with (capture / "golden.npy").open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            reader = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}.get(version)
            if reader is None:
                raise ValueError("owned-result reference has an unsupported NPY version")
            shapes = [reader(stream)[0]]
    parent = json.loads((capture / "parent/capture_receipt.json").read_text())
    return _references(capture, parent, entry.function_type.outputs,
                       metadata=capture / "parent/meta.json", result_shapes=shapes)


def compare(capture, out, rtol, atol, comparison, word_results, reference=None):
    from t2m.tensor_storage import assert_tensor_close

    expected, members = references(capture, reference)
    word_results = Q.checked_comparison(rtol, atol, comparison, word_results, result_count=len(expected))
    signature = json.loads((out / "compiler/signature.json").read_text())
    with np.load(out / "outputs.npz", allow_pickle=False) as saved:
        if set(saved.files) != {f"out{i}" for i in range(len(expected))} or len(signature["outputs"]) != len(expected):
            raise ValueError("owned-result target/reference rosters differ")
        for i, (wanted, row) in enumerate(zip(expected, signature["outputs"], strict=True)):
            actual = saved[f"out{i}"]
            if (actual.shape, actual.dtype) != (wanted.shape, wanted.dtype):
                raise ValueError("owned-result target/reference shape or storage differs")
            if comparison == "words" or i in word_results:
                if actual.tobytes() != wanted.tobytes():
                    raise ValueError("owned-result target differs from complete reference words")
            else:
                assert_tensor_close(actual, wanted, row["dtype"], rtol=rtol, atol=atol)
    return {"reference_comparison": "passed", "result_count": len(expected), "reference_members": members,
            "comparison": comparison, "word_results": list(word_results), "rtol": rtol, "atol": atol}


def qualify(capture, out, *, merlin_root, python, rtol, atol, features, comparison, word_results, inputs, reference):
    from t2m.bundle import verify_bundle
    from t2m.model_runtime_inputs import prepare

    capture, out, root = Path(capture).resolve(), Path(out).absolute(), Path(merlin_root).resolve()
    if out.exists() or out.is_symlink() or capture in out.parents:
        raise ValueError("owned-result target output must be new and outside the source bundle")
    receipt = verify_bundle(capture)
    kernel = receipt["schema"] == "t2m.capture-receipt.v2"
    if receipt["schema"] not in {"t2m.capture-receipt.v2", "t2m.bound-model-receipt.v1", "t2m.bound-model-receipt.v2"}:
        raise ValueError("owned-result target execution needs a complete supported source package")
    if (inputs is None) != (reference is None):
        raise ValueError("changed owned-result inputs and complete references must be supplied together")
    invocation = None
    if inputs is not None:
        inputs, reference = Path(inputs).absolute(), Path(reference).absolute()
        if any(path.is_symlink() or not path.is_file() for path in (inputs, reference)):
            raise ValueError("owned-result invocation archives must be direct regular files")
        invocation = {key: {"path": str(path.resolve()), **Q.pin(path)}
                      for key, path in (("inputs", inputs), ("reference", reference))}
    Q.checked_comparison(rtol, atol, comparison, word_results, result_count=len(references(capture, reference)[0]))
    original = Q.pin(capture / "capture_receipt.json")
    out.mkdir(parents=True)
    (out / "input-receipt.json").write_bytes((capture / "capture_receipt.json").read_bytes())
    Q.run_worker("stage", root, out, [capture, "execution-owned"], python)
    signature = json.loads((out / "compiler/signature.json").read_text())
    runtime_inputs = inputs or (capture / "inputs.npz" if kernel else prepare(capture, out, signature))
    Q.run_worker("build", root, out, [runtime_inputs, json.dumps(sorted(set(features) | {"reuse_loop_tensor_destinations"}))], python)
    Q.run_worker("run", root, out, [], python)
    evaluation = compare(capture, out, rtol, atol, comparison, word_results, reference)
    if verify_bundle(capture) != receipt or Q.pin(capture / "capture_receipt.json") != original:
        raise ValueError("owned-result source changed during target qualification")
    Q.write_json(out / "evaluation.json", evaluation)
    report = {"schema": SCHEMA, "status": "passed", "result_mode": "execution-owned",
        "source_directory": str(capture), "source_receipt": original, "package_kind": "kernel" if kernel else "model",
        "execution_scope": "functional-riscv-complete-execution-owned-results", "evaluation": evaluation,
        **({"invocation": invocation} if invocation else {}),
        "artifacts": {str(path.relative_to(out)): Q.pin(path) for path in sorted(out.rglob("*")) if path.is_file()}}
    Q.write_json(out / "qualification.json", report)
    try:
        verify(out, merlin_root=root, python=python)
    except BaseException:
        (out / "qualification.json").unlink(missing_ok=True)
        raise
    return report


def verify(out, *, merlin_root, python):
    from t2m.bundle import verify_bundle
    from t2m.model_runtime_inputs import verify as verify_inputs

    out = Path(out).absolute()
    report = json.loads((out / "qualification.json").read_text())
    if (report.get("schema") != SCHEMA or report.get("status") != "passed"
            or report.get("result_mode") != "execution-owned"
            or report.get("execution_scope") != "functional-riscv-complete-execution-owned-results"):
        raise ValueError("not a completed owned-result target qualification")
    required = {"input-receipt.json", "compiler/compile-inputs.json", "compiler/program.mlir", "compiler/signature.json",
                "image/model.elf", "image/cgen/driver.json", "build.json", "run.json", "target-process.json",
                "stdout.bin", "stderr.bin", "outputs.npz", "evaluation.json"}
    if not required <= set(report["artifacts"]):
        raise ValueError("incomplete owned-result target artifacts")
    if {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()} != \
            set(report["artifacts"]) | {"qualification.json"}:
        raise ValueError("owned-result target contains unaccounted artifacts")
    for name, value in report["artifacts"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or name == "qualification.json":
            raise ValueError("invalid owned-result target artifact path")
        if any(out.joinpath(*relative.parts[:i]).is_symlink() for i in range(len(relative.parts) + 1)) or Q.pin(out / name) != value:
            raise ValueError("owned-result target artifact changed: " + name)
    capture = Path(report["source_directory"])
    source_receipt = verify_bundle(capture)
    if (Q.pin(capture / "capture_receipt.json") != report["source_receipt"]
            or Q.pin(out / "input-receipt.json") != report["source_receipt"]):
        raise ValueError("owned-result target does not bind its unchanged parent package")
    if (source_receipt["schema"] == "t2m.capture-receipt.v2") != (report["package_kind"] == "kernel"):
        raise ValueError("owned-result target package kind differs from its source")
    stage = Q.checked_stage(out / "compiler")
    Q.checked_stage(out / "image-input", model_alias=True)
    if (stage["source"]["capture_receipt_sha256"] != report["source_receipt"]["sha256"]
            or Q.pin(out / "compiler/program.mlir") != Q.pin(out / "image-input/model.mlir")
            or Q.pin(out / "compiler/compile-inputs.json") != Q.pin(out / "image-input/compile-inputs.json")):
        raise ValueError("owned-result staged source/projection differs from its source package")
    build = json.loads((out / "build.json").read_text())
    signature = json.loads((out / "compiler/signature.json").read_text())
    invocation, reference = report.get("invocation"), None
    if invocation is not None:
        if set(invocation) != {"inputs", "reference"}:
            raise ValueError("owned-result invocation omits a source/reference archive")
        for row in invocation.values():
            path = Path(row["path"])
            if not path.is_absolute() or path.is_symlink() or Q.pin(path) != {key: row[key] for key in ("bytes", "sha256")}:
                raise ValueError("owned-result changed invocation archive differs from its binding")
        runtime_inputs, reference = Path(invocation["inputs"]["path"]), Path(invocation["reference"]["path"])
    else:
        runtime_inputs = capture / "inputs.npz" if report["package_kind"] == "kernel" else verify_inputs(capture, out, signature)
    if (signature.get("schema") != "merlin.compile-signature.v6"
            or build["source_program"] != Q.pin(out / "compiler/program.mlir")
            or build["runtime_input_path"] != str(runtime_inputs.resolve()) or build["runtime_inputs"] != Q.pin(runtime_inputs)
            or build["reference_values_supplied_to_build"] is not False
            or build["compiler_directory_contains_evaluator_data"] is not False):
        raise ValueError("owned-result source/input/compiler separation is inconsistent")
    for mode in ("stage", "build", "run"):
        if json.loads((out / f"{mode}-process.json").read_text())["returncode"] != 0:
            raise ValueError("owned-result target contains a failed qualification process")
    process = subprocess.run([str(python), str(Path(__file__).with_name("qualify_spike_model.py")),
        "--worker", "check", str(Path(merlin_root).resolve()), str(out)],
        capture_output=True, timeout=30, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(out)))
    if process.returncode:
        raise ValueError("owned-result retained target wire failed: " + process.stderr[-3000:].decode(errors="replace"))
    expected = compare(capture, out, report["evaluation"]["rtol"], report["evaluation"]["atol"],
        report["evaluation"]["comparison"], report["evaluation"]["word_results"], reference)
    if report["evaluation"] != expected or json.loads((out / "evaluation.json").read_text()) != expected:
        raise ValueError("owned-result retained evaluation differs from complete original references")
    return report
