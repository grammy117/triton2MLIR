"""Private subprocess entry point; optional native dependencies stay isolated."""

import ctypes
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import resource
import sys

import numpy as np
from torch_mlir import execution_engine, ir, passmanager, runtime
from runtime_shapes import resolve_shapes
from tensor_storage import DTYPES

SCALAR_CTYPES = {"f32": ctypes.c_float, "f64": ctypes.c_double, "i8": ctypes.c_int8,
                 "i16": ctypes.c_int16, "i32": ctypes.c_int32, "i64": ctypes.c_int64, "i1": ctypes.c_bool}


def main(work):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from diagnostic_runtime import prepare_stdout
    prepare_stdout()
    resource.setrlimit(resource.RLIMIT_DATA, (512 * 1024 * 1024, 512 * 1024 * 1024))
    work = Path(work)
    with np.load(work / "inputs.npz", allow_pickle=False) as saved:
        arrays = [saved[f"in{i}"].copy() for i in range(len(saved.files))]
    with ir.Context() as context:
        context.allow_unregistered_dialects = False
        module = ir.Module.parse((work / "model.mlir").read_text())
        functions = list(module.body.operations)
        if len(functions) != 1 or functions[0].operation.name != "func.func":
            raise ValueError("native runner requires a single standard entry")
        function = functions[0]
        signature = ir.FunctionType(ir.TypeAttr(function.attributes["function_type"]).value)
        if len(arrays) != len(signature.inputs):
            raise ValueError("native input count differs from function ABI")
        contract = json.loads((work / "shape-abi.json").read_text()) if (work / "shape-abi.json").exists() else None
        resolved = resolve_shapes(contract, arrays) if contract is not None else None
        for i, (array, type_) in enumerate(zip(arrays, signature.inputs, strict=True)):
            if isinstance(type_, ir.RankedTensorType):
                tensor = ir.RankedTensorType(type_)
                shape, dtype = tuple(tensor.shape), DTYPES[str(tensor.element_type)]
            elif str(type_) in SCALAR_CTYPES:
                shape, dtype = (), DTYPES[str(type_)]
            else:
                raise ValueError("native input has an unsupported scalar/tensor ABI type")
            if contract is not None:
                declared = contract["inputs"][i]
                symbolic = declared["shape"]
                if len(symbolic) != len(shape) or any((n < 0) != isinstance(d, str) or n >= 0 and n != d for n, d in zip(shape, symbolic)):
                    raise ValueError("native shape contract differs from its typed input signature")
            if (len(array.shape) != len(shape) or any(n >= 0 and n != actual for n, actual in zip(shape, array.shape))
                    or array.dtype != dtype or contract is None and any(n < 0 for n in shape)):
                raise ValueError("native input shape/dtype differs from the tensor ABI")
        shapes = tuple(tuple(ir.RankedTensorType(t).shape) for t in signature.results)
        if contract is not None:
            if len(contract["outputs"]) != len(shapes) or len(resolved) != len(shapes):
                raise ValueError("native shape contract differs from its output count")
            for shape, declared in zip(shapes, contract["outputs"], strict=True):
                if len(shape) != len(declared["shape"]) or any((n < 0) != isinstance(d, str) or n >= 0 and n != d for n, d in zip(shape, declared["shape"])):
                    raise ValueError("native shape contract differs from its typed output signature")
            shapes = resolved
        if any(n < 0 for shape in shapes for n in shape):
            raise ValueError("dynamic results require explicit runtime shape bindings")
        dtypes = [DTYPES[str(ir.RankedTensorType(t).element_type)] for t in signature.results]
        guard_size = 0
        guard_storage = 0
        if "prov.memory_guard_scratch_bytes" in function.attributes:
            guard_storage = ir.IntegerAttr(function.attributes["prov.memory_guard_scratch_bytes"]).value
            if guard_storage <= 0:
                raise ValueError("invalid native model guard storage contract")
        if "prov.memory_guard_bytes_per_result_element" in function.attributes:
            guard_size = ir.IntegerAttr(function.attributes["prov.memory_guard_bytes_per_result_element"]).value
            if guard_size not in (16, 24, 32):
                raise ValueError("unknown native memory guard storage contract")
            if sum(math.prod(shape) * (np.dtype(dtype).itemsize + guard_size) for shape, dtype in zip(shapes, dtypes, strict=True)) > 64 * 1024 * 1024:
                raise ValueError("native result/guard storage exceeds the 64MiB budget")
        if guard_storage + sum(math.prod(shape) * np.dtype(dtype).itemsize for shape, dtype in zip(shapes, dtypes, strict=True)) > 64 * 1024 * 1024:
            raise ValueError("native results exceed the 64MiB budget")
        outputs = [np.zeros(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes, strict=True)]
        name = ir.StringAttr(function.attributes["sym_name"]).value
        function.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        adapters = []
        def operations(op):
            yield op
            for region in op.regions:
                for block in region.blocks:
                    for item in block.operations:
                        yield from operations(item.operation)
        if any(op.name == "memref.generic_atomic_rmw" for op in operations(module.operation)):
            root = os.environ.get("T2M_MERLIN_ROOT")
            if not root:
                raise ValueError("native generic atomics require the explicit T2M_MERLIN_ROOT compiler support path")
            sys.path.insert(0, str(Path(root).resolve() / "src"))
            from merlin.llvmlower import atomic_compare_exchange
            namespace = {"ir": ir}
            exec(atomic_compare_exchange.SUPPORT, namespace)
            word_views = any(op.name == "memref.generic_atomic_rmw" and str(op.results[0].type) in ("f16", "f32", "f64")
                for op in operations(module.operation))
            count = namespace["_cas_rewrite"](context, module, word_views=word_views)
            prepared = work / ("prepared-atomic-word-views.mlir" if word_views else "prepared-compare-exchange.mlir")
            prepared.write_text(str(module))
            import hashlib
            adapters.append({"name": "merlin-exact-atomic-word-views" if word_views else "merlin-exact-integer-compare-exchange", "events": count,
                "support_form": "embedded-native-python-prelude",
                "support_sha256": hashlib.sha256(atomic_compare_exchange.SUPPORT.encode()).hexdigest(),
                "prepared_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest()})
            namespace["_cas_suspend"](module)
        has_bfloat = any("bf16" in str(value.type) for op in operations(module.operation) for value in (*op.operands, *op.results))
        pipelines = (
            "builtin.module(scf-forall-to-for,canonicalize,cse,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs-from-loops},buffer-results-to-out-params{modify-public-functions}" + (",buffer-deallocation-pipeline" if guard_size or guard_storage else "") + ")",
            "builtin.module(func.func(convert-linalg-to-loops)," + ("arith-expand{include-bf16=true}," if has_bfloat else "") + "expand-strided-metadata,lower-affine,convert-scf-to-cf)",
            "builtin.module(convert-math-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)",
        )
        for pipeline in pipelines:
            passmanager.PassManager.parse(pipeline).run(module.operation)
            if adapters and pipeline == pipelines[0]:
                namespace["_cas_restore"](module)
            if pipeline == pipelines[-1] and any(str(value.type) == "bf16" for op in operations(module.operation)
                    for value in (*op.operands, *op.results)):
                root = os.environ.get("T2M_MERLIN_ROOT")
                if not root:
                    raise ValueError("BF16 native transport requires the explicit T2M_MERLIN_ROOT compiler support path")
                sys.path.insert(0, str(Path(root).resolve() / "src"))
                from merlin.llvmlower import bfloat_transport
                scope = {"ir": ir}
                exec(bfloat_transport.SUPPORT, scope)
                count = scope["_bf_transport"](module)
                import hashlib
                adapters.append({"name": "merlin-exact-bfloat-transport", "scalar_values": count,
                    "arithmetic_casts": "upstream-arith-expand-include-bf16=true",
                    "support_sha256": hashlib.sha256(bfloat_transport.SUPPORT.encode()).hexdigest()})
        (work / "lowered-llvm.mlir").write_text(str(module))

        def check(operation):
            if operation.name.split(".")[0] not in ("builtin", "llvm") or operation.name == "builtin.unrealized_conversion_cast":
                raise ValueError("native execution legalization left " + operation.name)
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.operations:
                        check(child.operation)
        check(module.operation)
        module.operation.verify()
        engine = execution_engine.ExecutionEngine(module, opt_level=0)

        # MLIR's generic copy ABI uses two unranked strided descriptors. A small
        # host callback avoids requiring a separately built runner-utils library.
        class Unranked(ctypes.Structure):
            _fields_ = [("rank", ctypes.c_int64), ("descriptor", ctypes.c_void_p)]

        class Header(ctypes.Structure):
            _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                        ("offset", ctypes.c_int64)]

        callback_errors = []

        @ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.POINTER(Unranked), ctypes.POINTER(Unranked))
        def copy(element_bytes, source, destination):
            try:
                source, destination = source.contents, destination.contents
                rank = source.rank
                if rank != destination.rank or not 0 <= rank <= 8 or not 1 <= element_bytes <= 16:
                    raise ValueError("unexpected native copy descriptor")
                headers = [Header.from_address(d.descriptor) for d in (source, destination)]
                shapes = [(ctypes.c_int64 * rank).from_address(d.descriptor + ctypes.sizeof(Header)) for d in (source, destination)]
                strides = [(ctypes.c_int64 * rank).from_address(d.descriptor + ctypes.sizeof(Header) + rank * 8) for d in (source, destination)]
                if tuple(shapes[0]) != tuple(shapes[1]) or np.prod(shapes[0], dtype=np.int64) > 1_000_000:
                    raise ValueError("native copy shape mismatch or budget exceeded")
                for index in np.ndindex(tuple(shapes[0])):
                    addresses = [h.aligned + element_bytes * (h.offset + sum(i * s for i, s in zip(index, stride, strict=True)))
                                 for h, stride in zip(headers, strides, strict=True)]
                    ctypes.memmove(addresses[1], addresses[0], element_bytes)
            except Exception as exc:
                callback_errors.append(str(exc))

        engine.raw_register_runtime("memrefCopy", ctypes.cast(copy, ctypes.c_void_p))
        packed = []
        for array, type_ in zip(arrays, signature.inputs, strict=True):
            if isinstance(type_, ir.RankedTensorType):
                packed.append(ctypes.pointer(ctypes.pointer(runtime.get_ranked_memref_descriptor(array))))
            else:
                packed.append(ctypes.pointer(SCALAR_CTYPES[str(type_)](array.item())))
        packed.extend(ctypes.pointer(ctypes.pointer(runtime.get_ranked_memref_descriptor(a))) for a in outputs)
        observed = not signature.results or (work / "observe-config.json").is_file()
        if observed:
            from process_outcome import started, record
            started(work,backend="native-jit",program=work / "model.mlir",arguments=work / "inputs.npz",
                implementation={"kind":"llvm-jit","artifact":record(work / "lowered-llvm.mlir")})
            before = [a.tobytes() for a in arrays]
        engine.invoke(name, *packed)
        if observed and any(a.tobytes() != original for a,original in zip(arrays,before,strict=True)):
            raise ValueError("native effect-only entry mutated a readonly input")
        if callback_errors:
            raise ValueError("native runtime copy failed: " + "; ".join(callback_errors))
    np.savez(work / "outputs.npz", **{f"out{i}": a for i, a in enumerate(outputs)})
    (work / "execution.json").write_text(json.dumps({"schema": "t2m.native-execution.v1",
        "native_mlir_verified": True, "llvm_lowering_verified": True,
        "execution": "native-host-LLVM-JIT", "provider": "torch-mlir",
        "provider_version": version("torch-mlir"), "runtime": "bounded-generic-strided-copy",
        "source_triton_executed": False, "consumer_execution": "not-performed",
        **({"compiler_adapters": adapters} if adapters else {})}, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
