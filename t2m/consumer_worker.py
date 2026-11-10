"""Separate compiler-only Merlin materialization and evaluator-only invocation."""

import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys

COMPILER_FEATURES = frozenset({"reuse_loop_tensor_destinations"})


def tensor_pipeline(*, compare_exchange=False, word_views=False):
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
    passes = edit_pipeline(passes)
    if word_views:
        from merlin.llvmlower.atomic_word_views import edit_pipeline as edit_word_views
        passes = edit_word_views(passes)
    elif compare_exchange:
        from merlin.llvmlower.atomic_compare_exchange import edit_pipeline as edit_compare_exchange
        passes = edit_compare_exchange(passes)
    result = _splice(passes)
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
    features = COMPILER_FEATURES
    regions = [op for op in module.walk() if op.name == "memref.generic_atomic_rmw"]
    if any(str(op.results[0].type) in ("f16", "f32", "f64") for op in regions):
        features = features | {"lower_atomic_word_views"}
    elif regions:
        features = features | {"lower_integer_compare_exchange"}
    pipeline = tensor_pipeline(compare_exchange="lower_integer_compare_exchange" in features, word_views="lower_atomic_word_views" in features)
    llvm = lower_to_llvm_ir(prepared, workdir=work / "lowering", pipeline=pipeline,
                            timeout=30, features=features, retain_llvm_dialect=True)
    ll = work / "model.ll"
    ll.write_text(llvm)
    library = build_host_shared(ll, work / "model.so")
    report = {"schema": "t2m.merlin-host-compilation.v1", "compiler_input_staging": "passed",
              "whole_body_lowering": "passed", "native_object_and_link": "passed",
              "source_program": record(staged / "program.mlir"), "prepared_program": record(work / "prepared.mlir"),
              "llvm": record(ll), "shared_library": record(library), "pipeline": pipeline,
              "features": sorted(features),
              "lowering_api": "merlin.llvmlower.pipeline.lower_to_llvm_ir",
              "codegen_api": "merlin.llvmlower.codegen.build_host_shared",
              "compiler_python": str(compiler_python()), "codegen_tool": str(clang()),
              "host_machine": platform.machine(), "evaluator_data_supplied_to_compiler": False,
              "accelerator_execution": "not-performed"}
    (work / "compilation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def execute_model(staged, work, arguments, *, evaluation=None):
    import numpy as np
    from merlin.llvmlower.abi import HostModel, ScalarArg
    from runtime_shapes import resolve_shapes

    signature = json.loads((staged / "signature.json").read_text())
    evaluation = Path(evaluation) if evaluation is not None else work
    dtypes = {"f16": np.float16, "f32": np.float32, "f64": np.float64, "i8": np.int8, "i16": np.int16,
              "i32": np.int32, "i64": np.int64, "i1": np.bool_}
    with np.load(arguments, allow_pickle=False) as values:
        if set(values.files) != {f"in{i}" for i in range(len(signature["inputs"]))}:
            raise ValueError("Merlin host evaluator needs exactly the declared argument keys")
        arrays = [values[f"in{i}"].copy() for i in range(len(signature["inputs"]))]
    if signature["schema"] == "merlin.compile-signature.v2":
        shapes = resolve_shapes({"schema": "merlin.runtime-shapes.v1", "inputs": signature["inputs"], "outputs": signature["outputs"]}, arrays)
    elif signature["schema"] == "merlin.compile-signature.v1":
        shapes = [value["shape"] for value in signature["outputs"]]
    else:
        raise ValueError("unsupported Merlin invocation signature")
    for array, declared in zip(arrays, signature["inputs"], strict=True):
        if (signature["schema"] == "merlin.compile-signature.v1" and list(array.shape) != declared["shape"]
                or len(array.shape) != len(declared["shape"]) or any(type(n) is int and n != actual for n, actual in zip(declared["shape"], array.shape))
                or array.dtype != dtypes[declared["dtype"]]):
            raise ValueError("Merlin host evaluator argument differs from staged ABI")
        if declared.get("kind") == "scalar" and "bounds" in declared:
            if not declared["bounds"][0] <= array.item() <= declared["bounds"][1]:
                raise ValueError("Merlin host runtime scalar is outside its guarded ABI bounds")
    guard_size = signature.get("guard_storage_bytes_per_result_element", 0)
    guard_storage = signature.get("guard_storage_bytes", 0)
    if type(guard_storage) is not int or guard_storage < 0:
        raise ValueError("invalid Merlin model guard storage contract")
    if guard_size not in (0, 16, 24, 32):
        raise ValueError("unknown Merlin memory guard storage contract")
    if guard_storage + sum(a.nbytes for a in arrays) + sum(math.prod(shape) * (np.dtype(dtypes[value["dtype"]]).itemsize + guard_size)
                                         for shape, value in zip(shapes, signature["outputs"], strict=True)) > 128 * 1024 * 1024:
        raise ValueError("Merlin host invocation exceeds its 128MiB array budget")
    outputs = [np.zeros(shape, dtype=dtypes[value["dtype"]]) for shape, value in zip(shapes, signature["outputs"], strict=True)]
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
