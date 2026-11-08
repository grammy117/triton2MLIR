"""Private subprocess entry point; optional native dependencies stay isolated."""

import ctypes
from importlib.metadata import version
import json
from pathlib import Path
import resource
import sys

import numpy as np
from torch_mlir import execution_engine, ir, passmanager, runtime

DTYPES = {"f32": np.float32, "f64": np.float64, "i8": np.int8,
          "i16": np.int16, "i32": np.int32, "i64": np.int64}


def main(work):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
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
        for array, type_ in zip(arrays, signature.inputs, strict=True):
            tensor = ir.RankedTensorType(type_)
            dtype = DTYPES[str(tensor.element_type)]
            if array.shape != tuple(tensor.shape) or array.dtype != dtype:
                raise ValueError("native input shape/dtype differs from the tensor ABI")
        outputs = [np.zeros(ir.RankedTensorType(t).shape, dtype=DTYPES[str(ir.RankedTensorType(t).element_type)])
                   for t in signature.results]
        if sum(x.nbytes for x in outputs) > 64 * 1024 * 1024:
            raise ValueError("native results exceed the 64MiB budget")
        name = ir.StringAttr(function.attributes["sym_name"]).value
        function.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        pipelines = (
            "builtin.module(scf-forall-to-for,canonicalize,cse,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},buffer-results-to-out-params{modify-public-functions})",
            "builtin.module(func.func(convert-linalg-to-loops),lower-affine,convert-scf-to-cf,expand-strided-metadata,convert-math-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)",
        )
        for pipeline in pipelines:
            passmanager.PassManager.parse(pipeline).run(module.operation)

        def check(operation):
            if operation.name.split(".")[0] not in ("builtin", "llvm") or operation.name == "builtin.unrealized_conversion_cast":
                raise ValueError("native execution legalization left " + operation.name)
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.operations:
                        check(child.operation)
        check(module.operation)
        module.operation.verify()
        (work / "lowered-llvm.mlir").write_text(str(module))
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
        descriptors = [runtime.get_ranked_memref_descriptor(a) for a in [*arrays, *outputs]]
        engine.invoke(name, *(ctypes.pointer(ctypes.pointer(d)) for d in descriptors))
        if callback_errors:
            raise ValueError("native runtime copy failed: " + "; ".join(callback_errors))
    np.savez(work / "outputs.npz", **{f"out{i}": a for i, a in enumerate(outputs)})
    (work / "execution.json").write_text(json.dumps({"schema": "t2m.native-execution.v1",
        "native_mlir_verified": True, "llvm_lowering_verified": True,
        "execution": "native-host-LLVM-JIT", "provider": "torch-mlir",
        "provider_version": version("torch-mlir"), "runtime": "bounded-generic-strided-copy",
        "source_triton_executed": False, "consumer_execution": "not-performed"}, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
