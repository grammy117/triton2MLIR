"""Separate compiler-only Merlin materialization and evaluator-only invocation."""

import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys

COMPILER_FEATURES = frozenset({"reuse_loop_tensor_destinations"})


def tensor_pipeline(*, compare_exchange=False, word_views=False, integer_regions=False, bfloat=False, bfloat_elementwise=False, fp8_widening=False, fp8_narrowing=False, literal_diagnostics=False, scalar_diagnostics=False):
    # Reuse Merlin's real scalar pipeline, including allocation ownership.
    # The adapter adds logical-grid serialization at this selected host backend
    # and allows bufferized loop results. Checked destination reuse runs after
    # fusion; emitted frontend IR retains the source tensor recurrence.
    from merlin.llvmlower.pipeline import _UPSTREAM_PASSES, _splice
    from merlin.llvmlower.loop_tensor_destinations import edit_pipeline

    passes, adapted = ["scf-forall-to-for"], 0
    for entry in _UPSTREAM_PASSES:
        if bfloat and entry == "convert-arith-to-llvm":
            # Official bit expansion preserves BF16 subnormal casts even when
            # a host BF16 conversion instruction would flush them to zero.
            passes.append("arith-expand{include-bf16=true}")
        if entry.startswith("one-shot-bufferize{") and entry.endswith("}"):
            entry = entry[:-1] + " allow-return-allocs-from-loops}"
            adapted += 1
        passes.append(entry)
    passes = edit_pipeline(passes)
    if scalar_diagnostics:
        from merlin.llvmlower.scalar_diagnostics import edit_pipeline as edit_scalar_diagnostics
        passes = edit_scalar_diagnostics(passes)
    elif literal_diagnostics:
        from merlin.llvmlower.length_aware_diagnostics import edit_pipeline as edit_literal_diagnostics
        passes = edit_literal_diagnostics(passes)
    if bfloat:
        from merlin.llvmlower.bfloat_transport import edit_pipeline as edit_bfloat_transport
        passes = edit_bfloat_transport(passes)
    if word_views:
        from merlin.llvmlower.atomic_word_views import edit_pipeline as edit_word_views
        passes = edit_word_views(passes)
    elif compare_exchange:
        from merlin.llvmlower.atomic_compare_exchange import edit_pipeline as edit_compare_exchange
        passes = edit_compare_exchange(passes)
    if integer_regions:
        from merlin.llvmlower.atomic_integer_regions import edit_pipeline as edit_integer_regions
        passes = edit_integer_regions(passes)
    if bfloat_elementwise:
        from merlin.llvmlower.bfloat_elementwise import edit_pipeline as edit_bfloat_elementwise
        passes = edit_bfloat_elementwise(passes)
    if fp8_widening:
        from merlin.llvmlower.fp8_widening import edit_pipeline as edit_fp8_widening
        passes = edit_fp8_widening(passes)
    if fp8_narrowing:
        from merlin.llvmlower.fp8_narrowing import edit_pipeline as edit_fp8_narrowing
        passes = edit_fp8_narrowing(passes)
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
    if any("bf16" in str(value.type) for op in module.walk() for value in (*op.operands, *op.results)):
        features = features | {"preserve_bfloat_transport"}
    bfloat_math = "emulate_bfloat_elementwise"
    from bfloat_math import requires_emulation
    if requires_emulation(module.walk()):
        features = features | {bfloat_math}
    from fp8_math import requires_widening,requires_narrowing
    if requires_widening(module.walk()):
        features = features | {"lower_fp8_widening"}
    if requires_narrowing(module.walk()):
        features = features | {"lower_fp8_narrowing"}
    if any(op.name == "vector.print" for op in module.walk()):
        features = features | {"lower_scalar_diagnostics"}
    elif any(op.name == "cf.assert" and "\x00" in op.msg.data for op in module.walk()):
        features = features | {"lower_length_aware_diagnostics"}
    regions = [op for op in module.walk() if op.name == "memref.generic_atomic_rmw"]
    from native import requires_integer_atomic_regions
    integer_regions = requires_integer_atomic_regions(module)
    if integer_regions:
        from merlin.llvmlower.atomic_integer_regions import ensure_registered
        ensure_registered()
        features = features | {"lower_integer_atomic_regions"}
    if any(str(op.results[0].type) in ("f16", "f32", "f64") for op in regions):
        features = features | {"lower_atomic_word_views"}
    elif regions and not integer_regions:
        features = features | {"lower_integer_compare_exchange"}
    pipeline = tensor_pipeline(compare_exchange="lower_integer_compare_exchange" in features, word_views="lower_atomic_word_views" in features, integer_regions=integer_regions,
        bfloat="preserve_bfloat_transport" in features, bfloat_elementwise=bfloat_math in features,
        fp8_widening="lower_fp8_widening" in features,
        fp8_narrowing="lower_fp8_narrowing" in features,
        literal_diagnostics="lower_length_aware_diagnostics" in features,
        scalar_diagnostics="lower_scalar_diagnostics" in features)
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


def execute_model(staged, work, arguments, *, evaluation=None, observed=False):
    from diagnostic_runtime import prepare_stdout
    prepare_stdout()
    import numpy as np
    from merlin.llvmlower.abi import HostModel, ScalarArg
    from runtime_shapes import resolve_shapes
    from tensor_storage import DTYPES as dtypes

    signature = json.loads((staged / "signature.json").read_text())
    evaluation = Path(evaluation) if evaluation is not None else work
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
    costs = signature.get("guard_storage_bytes_by_result_element", [guard_size] * len(shapes))
    if (not isinstance(costs, list) or len(costs) != len(shapes)
            or any(type(cost) is not int or cost not in {0, *(base * width for base in (16, 24, 32) for width in (1, 2, 4, 8))} for cost in costs)
            or max(costs, default=guard_size) != guard_size):
        raise ValueError("unknown Merlin memory guard storage contract")
    if guard_storage + sum(a.nbytes for a in arrays) + sum(math.prod(shape) * (np.dtype(dtypes[value["dtype"]]).itemsize + cost)
                                         for shape, value, cost in zip(shapes, signature["outputs"], costs, strict=True)) > 128 * 1024 * 1024:
        raise ValueError("Merlin host invocation exceeds its 128MiB array budget")
    outputs = [np.zeros(shape, dtype=dtypes[value["dtype"]]) for shape, value in zip(shapes, signature["outputs"], strict=True)]
    compiled = json.loads((work / "compilation.json").read_text())
    if record(work / "model.so") != compiled["shared_library"]:
        raise ValueError("Merlin shared library changed before invocation")
    model = HostModel.load(str((work / "model.so").resolve()))
    if observed:
        from process_outcome import started
        started(evaluation,backend="merlin-host",program=staged / "program.mlir",arguments=arguments,
            implementation={"kind":"shared-library","artifact":record(work / "model.so")})
    before = [a.tobytes() for a in arrays]
    first = None
    invocations = 1 if observed else 2
    for invocation in range(invocations):
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
        "invocations": invocations, "functional_inputs_preserved": True, "outputs": record(evaluation / "outputs.npz"),
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
    elif mode == "observe" and len(remaining) == 2:
        execute_model(staged, work, remaining[0], evaluation=remaining[1],observed=True)
    else:
        raise ValueError("expected a compiler-only compile or separate evaluator invocation")
