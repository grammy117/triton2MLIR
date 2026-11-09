"""Separate compiler-only Merlin materialization and evaluator-only invocation."""

import hashlib
import json
from pathlib import Path
import platform
import resource
import sys

COMPILER_FEATURES = frozenset({"reuse_loop_tensor_destinations"})


def tensor_pipeline():
    # Reuse Merlin's real scalar pipeline, including allocation ownership.
    # The adapter adds logical-grid serialization at this selected host backend
    # and allows bufferized loop results. Checked destination reuse runs after
    # fusion; emitted frontend IR retains the source tensor recurrence.
    from merlin.llvmlower.pipeline import _UPSTREAM_PASSES, _splice
    from merlin.llvmlower.loop_tensor_destinations import edit_pipeline

    passes, adapted = ["scf-forall-to-for"], 0
    for entry in _UPSTREAM_PASSES:
        if entry.startswith("one-shot-bufferize{") and entry.endswith("}"):
            entry = entry[:-1] + " allow-return-allocs-from-loops}"
            adapted += 1
        passes.append(entry)
    result = _splice(edit_pipeline(passes))
    if adapted != 1 or "ownership-based-buffer-deallocation" not in result:
        raise ValueError("Merlin scalar pipeline lacks the qualified loop/ownership boundary")
    return result


def record(path):
    data = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def compile_model(staged, work):
    # This entry receives only the staged compiler directory. It has no runtime
    # arrays, golden or evaluator argument path in its command line.
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import add_c_interface
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.codegen import build_host_shared
    from merlin.llvmlower.toolchain import clang, compiler_python
    from merlin.xdsl_dialects._common import text

    manifest = json.loads((staged / "compile-inputs.json").read_text())
    if manifest.get("schema") != "merlin.compile-inputs.v1":
        raise ValueError("expected staged Merlin compiler inputs")
    for name, digest in manifest["members"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or record(staged / path)["sha256"] != digest:
            raise ValueError("compiler input changed after staging")
    source = (staged / "program.mlir").read_text()
    module = parse_mlir_text(source)
    module.verify()
    add_c_interface(module)
    prepared = text(module, generic=True)
    (work / "prepared.mlir").write_text(prepared)
    pipeline = tensor_pipeline()
    llvm = lower_to_llvm_ir(prepared, workdir=work / "lowering", pipeline=pipeline,
                            timeout=30, features=COMPILER_FEATURES, retain_llvm_dialect=True)
    ll = work / "model.ll"
    ll.write_text(llvm)
    library = build_host_shared(ll, work / "model.so")
    report = {"schema": "t2m.merlin-host-compilation.v1", "compiler_input_staging": "passed",
              "whole_body_lowering": "passed", "native_object_and_link": "passed",
              "source_program": record(staged / "program.mlir"), "prepared_program": record(work / "prepared.mlir"),
              "llvm": record(ll), "shared_library": record(library), "pipeline": pipeline,
              "features": sorted(COMPILER_FEATURES),
              "lowering_api": "merlin.llvmlower.pipeline.lower_to_llvm_ir",
              "codegen_api": "merlin.llvmlower.codegen.build_host_shared",
              "compiler_python": str(compiler_python()), "codegen_tool": str(clang()),
              "host_machine": platform.machine(), "evaluator_data_supplied_to_compiler": False,
              "accelerator_execution": "not-performed"}
    (work / "compilation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def execute_model(staged, work, arguments, *, evaluation=None):
    import numpy as np
    from merlin.llvmlower.abi import HostModel, ScalarArg

    signature = json.loads((staged / "signature.json").read_text())
    evaluation = Path(evaluation) if evaluation is not None else work
    dtypes = {"f32": np.float32, "f64": np.float64, "i8": np.int8, "i16": np.int16,
              "i32": np.int32, "i64": np.int64, "i1": np.bool_}
    with np.load(arguments, allow_pickle=False) as values:
        if set(values.files) != {f"in{i}" for i in range(len(signature["inputs"]))}:
            raise ValueError("Merlin host evaluator needs exactly the declared argument keys")
        arrays = [values[f"in{i}"].copy() for i in range(len(signature["inputs"]))]
    for array, declared in zip(arrays, signature["inputs"], strict=True):
        if list(array.shape) != declared["shape"] or array.dtype != dtypes[declared["dtype"]]:
            raise ValueError("Merlin host evaluator argument differs from staged ABI")
        if declared.get("kind") == "scalar" and "bounds" in declared:
            if not declared["bounds"][0] <= array.item() <= declared["bounds"][1]:
                raise ValueError("Merlin host runtime scalar is outside its guarded ABI bounds")
    outputs = [np.zeros(value["shape"], dtype=dtypes[value["dtype"]]) for value in signature["outputs"]]
    if sum(a.nbytes for a in [*arrays, *outputs]) > 128 * 1024 * 1024:
        raise ValueError("Merlin host invocation exceeds its 128MiB array budget")
    compiled = json.loads((work / "compilation.json").read_text())
    if record(work / "model.so") != compiled["shared_library"]:
        raise ValueError("Merlin shared library changed before invocation")
    model = HostModel.load(str((work / "model.so").resolve()))
    before = [a.tobytes() for a in arrays]
    first = None
    for invocation in range(2):
        if invocation:
            for output in outputs:
                output.fill(np.nan if np.issubdtype(output.dtype, np.floating) else 85)
        invocation_args = [ScalarArg(array.item(), declared["dtype"]) if declared.get("kind") == "scalar"
                           else (array.ctypes.data, array.shape) for array, declared in zip(arrays, signature["inputs"], strict=True)]
        model([*invocation_args, *[(a.ctypes.data, a.shape) for a in outputs]])
        if any(array.tobytes() != original for array, original in zip(arrays, before, strict=True)):
            raise ValueError("Merlin compiled function mutated a functional tensor input")
        observed = [a.tobytes() for a in outputs]
        if first is not None and first != observed:
            raise ValueError("Merlin repeated invocation has stateful or incomplete output publication")
        first = observed
    np.savez(evaluation / "outputs.npz", **{f"out{i}": a for i, a in enumerate(outputs)})
    (evaluation / "evaluation.json").write_text(json.dumps({"schema": "t2m.merlin-host-evaluation.v1",
        "invocations": 2, "functional_inputs_preserved": True, "outputs": record(evaluation / "outputs.npz"),
        "arguments": record(arguments), "source_program": record(staged / "program.mlir"),
        "shared_library": record(work / "model.so")}, indent=2) + "\n")


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    mode, root, staged, work, *remaining = sys.argv[1:]
    sys.path.insert(0, str(Path(root) / "src"))
    staged, work = Path(staged), Path(work)
    if mode == "compile" and not remaining:
        compile_model(staged, work)
    elif mode == "execute" and len(remaining) == 1:
        execute_model(staged, work, remaining[0])
    elif mode == "invoke" and len(remaining) == 2:
        execute_model(staged, work, remaining[0], evaluation=remaining[1])
    else:
        raise ValueError("expected a compiler-only compile or separate evaluator invocation")
