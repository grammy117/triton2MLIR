import argparse
import json
from pathlib import Path

import numpy as np

from .api import convert
from .bundle import verify_bundle, write_bundle
from .spec import BufferSpec, KernelSpec


def demo(out, *, size=1024, block=256, operation="add"):
    if size <= 0 or block <= 0:
        raise ValueError("demo requires positive size and block")
    from .examples.vector_add import vector_add, multiply_add, masked_add, masked_multiply_add
    ragged = bool(size % block)
    kernel = ((masked_add if operation == "add" else masked_multiply_add) if ragged
              else (vector_add if operation == "add" else multiply_add))
    programs = (size + block - 1) // block
    constexprs = (("BLOCK", block),) + ((("N", size),) if ragged else ())
    spec = KernelSpec(kernel.__name__, tuple(BufferSpec(n, (size,), e) for n, e in
                     (("x_ptr", "read"), ("y_ptr", "read"), ("out_ptr", "write"))),
                     (programs,), constexprs,
                     "Elementwise vector sum" if operation == "add" else "Elementwise x*y+x")
    result = convert(kernel, spec)
    rng = np.random.default_rng(0)
    x, y = (rng.standard_normal(size, dtype=np.float32) for _ in range(2))
    golden = x + y if operation == "add" else x * y + x
    write_bundle(result, out, [x, y], golden)
    return {"bundle": str(out), "kernel": spec.entry, "elements": size, "programs": programs,
            "masked_tail": ragged,
            "source_operations": len(result.trace["source_operations"]),
            "numerical_check": "passed", "merlin_staged_execution": "not-certified"}


def shared_demo(out, *, kernel="softmax", optimizer=None, python=None, memory_profile="tensor", address_policy="structured"):
    from .examples import tensor_kernels as kernels
    from .frontend import capture_ttir
    from .shared_tensor import TensorABI, TensorBuffer, convert_shared
    from .shared_bundle import write_shared_bundle
    from .scalar_abi import RuntimeScalar
    from .dynamic_abi import GridExtent
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    work.mkdir(parents=True, exist_ok=False)
    grid = (1,)
    runtime_scalars = ()
    example_dimensions = {}
    buffer_effects = {}
    buffer_dtypes = {}
    if kernel == "tiled-add":
        fn, constants, shapes, grid = kernels.tiled_add, {"N": 16, "BM": 4, "BN": 8}, [(8, 16)] * 3, (2, 2)
    elif kernel == "tiled-matmul":
        fn, constants, shapes, grid = kernels.tiled_matmul, {"N": 32, "K": 16, "BM": 16, "BN": 16}, [(32, 16), (16, 32), (32, 32)], (2, 2)
    elif kernel == "loop-matmul":
        fn, constants, shapes, grid = kernels.loop_matmul, {"N": 32, "K": 32, "BM": 16, "BN": 16, "BK": 16}, [(32, 32)] * 3, (2, 2)
    elif kernel in ("ragged-matmul", "blockptr-matmul", "descriptor-matmul"):
        fn = {"ragged-matmul": kernels.ragged_matmul, "blockptr-matmul": kernels.blockptr_matmul,
              "descriptor-matmul": kernels.descriptor_matmul}[kernel]
        m, n, k = (35, 20, 36) if kernel == "descriptor-matmul" else (35, 21, 37)
        grid = (3, 2)
        constants = {"M": m, "N": n, "K": k, "BM": 16, "BN": 16, "BK": 16}
        if kernel == "ragged-matmul":
            constants["TRANSPOSED"] = False
        shapes = [(m, k), (k, n), (m, n)]
    elif kernel == "indirect-store":
        fn, constants = kernels.indirect_store, {"B": 8, "REPEAT": 3, "READ_OUTPUT": False, "STORE_INPUT": False}
        shapes, grid = [("n",), ("n",), ("m",)], (GridExtent("n", 8),)
        runtime_scalars = (RuntimeScalar("n", "i32", 3, bounds=(0, 31)), RuntimeScalar("m", "i32", 4, bounds=(0, 31)))
        example_dimensions = {"n": 13, "m": 23}
    elif kernel == "matrix-add":
        fn, constants, shapes = kernels.matrix_add, {"M": 4, "N": 8}, [(4, 8)] * 3
    elif kernel == "matmul":
        fn, constants, shapes = kernels.block_matmul, {"M": 16, "N": 16, "K": 16}, [(16, 16)] * 3
    elif kernel == "scan":
        fn, constants, shapes, grid = kernels.scan_tiles, {"N": 53, "B": 16, "KIND": 0, "REVERSE": True}, [(53,)] * 2, (4,)
    elif kernel == "tuple-scan":
        fn, constants, shapes = kernels.tuple_scan, {"R": 2, "S": 8, "AXIS": 1, "REVERSE": True}, [(2, 8)] * 4
    elif kernel == "indexed-gather":
        fn, constants, shapes, grid = kernels.indexed_gather, {"N": 13, "B": 8, "MASKED": True}, [(3, 5), (13,), (13,)], (2,)
    elif kernel == "runtime-affine":
        fn, constants, shapes, grid = kernels.runtime_affine, {"N": 13, "B": 8}, [(13,), (13,)], (2,)
        runtime_scalars = (RuntimeScalar("alpha", "f32", 1), RuntimeScalar("bias", "f32", 3))
    elif kernel == "runtime-prefix":
        fn, constants, shapes, grid = kernels.runtime_prefix, {"B": 8}, [(13,), (13,)], (4,)
        runtime_scalars = (RuntimeScalar("n", "i32", 1, bounds=(0, 13)),)
    elif kernel == "runtime-recurrence":
        fn, constants, shapes, grid = kernels.runtime_recurrence, {"N": 13, "B": 8}, [(13,), (13,)], (2,)
        runtime_scalars = (RuntimeScalar("iterations", "i32", 1, bounds=(0, 5)),)
    elif kernel == "runtime-memory-loop":
        fn, constants, shapes, grid = kernels.runtime_memory_loop, {"N": 13, "B": 8}, [(65,), (13,)], (2,)
        runtime_scalars = (RuntimeScalar("iterations", "i32", 1, bounds=(-2, 5)),)
    elif kernel == "pointer-while":
        from .examples.pointer_control_kernels import while_pointer_sum
        fn, constants, shapes = while_pointer_sum, {"B": 8}, [("rows", "n"), ("n",)]
        grid = (GridExtent("n", 8),)
        runtime_scalars = (RuntimeScalar("n", "i32", 2, bounds=(0, 13)),
                           RuntimeScalar("rows", "i32", 3, bounds=(1, 5)),
                           RuntimeScalar("steps", "i32", 4, bounds=(0, 4)))
        example_dimensions = {"n": 9, "rows": 3}
    elif kernel == "selected-memory":
        from .examples.selected_pointer_kernels import selected_access
        fn, constants = selected_access, {"B": 8, "SCALAR": False, "LANES": True, "STORE_SELECTED": True}
        shapes, grid = [("n",), ("m",), ("n",)], (GridExtent("n", 8),)
        runtime_scalars = (RuntimeScalar("n", "i32", 3, bounds=(0, 31)),
                           RuntimeScalar("m", "i32", 4, bounds=(0, 37)),
                           RuntimeScalar("choice", "i32", 5, bounds=(0, 3)))
        example_dimensions = {"n": 13, "m": 15}
        buffer_effects = {"x": "readwrite", "y": "readwrite", "out": "write"}
    elif kernel == "online-attention":
        from .examples.attention_kernels import online_attention
        fn, constants = online_attention, {"D": 16, "BM": 16, "BN": 16}
        shapes, grid = [("m", 16), ("n", 16), ("n", 16), ("m", 16)], (GridExtent("m", 16),)
        runtime_scalars = (RuntimeScalar("m", "i32", 4, bounds=(0, 35)), RuntimeScalar("n", "i32", 5, bounds=(0, 37)))
        example_dimensions = {"m": 17, "n": 19}
    elif kernel == "histogram":
        from .examples.histogram_kernels import histogram_tiles
        fn, constants = histogram_tiles, {"B": 16, "NB": 8, "MASKED": True, "ROW_VIEW": True, "ROW_MASK": True}
        shapes, grid = [("n",), ("tiles", 8)], (GridExtent("tiles", 1),)
        runtime_scalars = (RuntimeScalar("n", "i32", 2, bounds=(0, 31)), RuntimeScalar("tiles", "i32", 3, bounds=(0, 3)))
        example_dimensions = {"n": 23, "tiles": 2}
        buffer_effects, buffer_dtypes = {"out": "write"}, {"x": "i32", "out": "i32"}
    elif kernel == "runtime-memory-store":
        fn, constants, shapes, grid = kernels.runtime_memory_accumulate, {"N": 13, "B": 8}, [(13,), (13,)], (2,)
        runtime_scalars = (RuntimeScalar("iterations", "i32", 1, bounds=(-2, 5)),)
    elif kernel == "dynamic-count":
        fn, constants, shapes, grid = kernels.dynamic_count, {"B": 8}, [("n",)] * 2, (GridExtent("n", 8),)
        runtime_scalars = (RuntimeScalar("n", "i32", 1, bounds=(0, 13)),)
        example_dimensions = {"n": 7}
    elif kernel == "dynamic-matmul":
        fn, constants = kernels.dynamic_matmul, {"BM": 16, "BN": 16, "BK": 16, "TRANSPOSED": False}
        shapes, grid = [("m", "k"), ("k", "n"), ("m", "n")], (GridExtent("m", 16), GridExtent("n", 16))
        runtime_scalars = tuple(RuntimeScalar(name, "i32", arg, bounds=(0, high))
            for name, arg, high in (("m", 2, 35), ("n", 3, 21), ("k", 4, 37)))
        example_dimensions = {"m": 35, "n": 21, "k": 37}
    elif kernel == "dynamic-gather":
        fn, constants, shapes = kernels.dynamic_indexed_gather, {"B": 8}, [("m", "n"), ("count",), ("count",)]
        grid = (GridExtent("count", 8),)
        runtime_scalars = tuple(RuntimeScalar(name, "i32", arg, bounds=(0, high))
            for name, arg, high in (("m", 2, 7), ("n", 3, 11), ("count", 4, 19)))
        example_dimensions = {"m": 3, "n": 5, "count": 13}
    elif kernel == "masked-store":
        fn, constants, shapes = kernels.masked_tile_store, {"B": 8, "STEP": 8, "MODE": 2, "FRESH": False}, [("n",)] * 2
        grid, runtime_scalars = (GridExtent("n", 8),), (RuntimeScalar("n", "i32", 2, bounds=(0, 13)),)
        example_dimensions = {"n": 13}
    elif kernel == "masked-matrix":
        fn, constants = kernels.masked_matrix_store, {"BM": 4, "BN": 8, "TRANSPOSED": True, "TRAVERSE_TRANSPOSED": True, "BAD_MASK": False}
        shapes, grid = [("m", "n"), ("n", "m")], (GridExtent("m", 4), GridExtent("n", 8))
        runtime_scalars = (RuntimeScalar("m", "i32", 2, bounds=(0, 9)), RuntimeScalar("n", "i32", 3, bounds=(0, 13)))
        example_dimensions = {"m": 9, "n": 13}
    else:
        fn, constants, shapes = kernels.row_softmax, {"N": 7, "BLOCK": 8}, [(7,)] * 2
    names = ["x", "y", "out"] if len(shapes) == 3 else ["x", "out"]
    if kernel == "tuple-scan":
        names = ["a", "b", "out_a", "out_b"]
    elif kernel == "online-attention":
        names = ["q", "k", "v", "out"]
    elif kernel in ("indexed-gather", "dynamic-gather", "indirect-store"):
        names = ["x", "indices", "out"]
    output_effect = "write" if kernel in ("ragged-matmul", "blockptr-matmul", "descriptor-matmul", "scan", "tuple-scan", "indexed-gather", "runtime-affine", "runtime-recurrence", "runtime-memory-loop", "pointer-while", "online-attention", "dynamic-count", "dynamic-matmul", "dynamic-gather") else "readwrite"
    abi = TensorABI(fn.__name__, tuple(TensorBuffer(n, s, buffer_effects.get(n, output_effect if n.startswith("out") else "read"),
                    dtype=buffer_dtypes.get(n, "i32" if n == "indices" else "f32"))
                    for n, s in zip(names, shapes, strict=True)), grid=grid, intent=f"Source {kernel} example", runtime_scalars=runtime_scalars)
    source = work / "kernel.ttir"
    source.write_text(capture_ttir(fn, {**{b.name: "*fp32" if b.dtype == "f32" else "*" + b.dtype for b in abi.buffers},
                                      **{s.name: "fp32" if s.dtype == "f32" else s.dtype for s in abi.runtime_scalars},
                                      **{n: "constexpr" for n in constants}}, constants))
    result = convert_shared(source, abi, work / "translation", optimizer=optimizer, memory_profile=memory_profile, address_policy=address_policy)
    rng = np.random.default_rng(7)
    arrays = [rng.normal(size=tuple(example_dimensions.get(n, n) for n in b.shape)).astype("float32") for b in abi.inputs]
    additional = []
    if kernel in ("matrix-add", "tiled-add"):
        golden = arrays[0] + arrays[1]
    elif kernel == "indirect-store":
        arrays[1] = np.array([17, 5, 11, -99, 3, 8, 7, 16, 0, 14, 12, -1, 6], dtype=np.int32)
        arrays[3:] = [np.asarray(example_dimensions[name], dtype=np.int32) for name in ("n", "m")]
        golden = arrays[2].copy()
        active = arrays[1] >= 0
        golden[arrays[1][active]] = (arrays[0][active] + np.float32(2)) + np.float32(1)
    elif kernel == "runtime-affine":
        arrays[1:] = [np.asarray(1.5, dtype=np.float32), np.asarray(-3, dtype=np.float32)]
        golden = arrays[0] * arrays[1] + arrays[2]
    elif kernel == "runtime-prefix":
        arrays[2] = np.asarray(7, dtype=np.int32)
        golden = arrays[1].copy()
        golden[:7] = arrays[0][:7] + 2
    elif kernel == "runtime-recurrence":
        arrays[1] = np.asarray(3, dtype=np.int32)
        golden = arrays[0].copy()
        for k in range(3):
            golden = golden * np.float32(0.5) + np.float32(k)
    elif kernel == "runtime-memory-loop":
        arrays[1] = np.asarray(3, dtype=np.int32)
        golden = np.zeros(13, dtype=np.float32)
        for k in range(3):
            golden += arrays[0][k * 13:(k + 1) * 13]
    elif kernel == "pointer-while":
        arrays[1:] = [np.asarray(value, dtype=np.int32) for value in
                      (example_dimensions["n"], example_dimensions["rows"], example_dimensions["rows"] - 1)]
        golden = arrays[0].sum(axis=0)
    elif kernel == "selected-memory":
        n, m, choice = example_dimensions["n"], example_dimensions["m"], 1
        arrays[2:] = [np.asarray(value, dtype=np.int32) for value in (n, m, choice)]
        index = np.arange(n)
        take_x = (index + choice) % 2 == 0
        golden, y = arrays[0] + np.float32(3), arrays[1].copy()
        golden[take_x] *= np.float32(2)
        y[m - 1 - index[~take_x]] *= np.float32(2)
        selected = golden.copy()
        selected[~take_x] = y[m - 1 - index[~take_x]]
        additional = [y, selected]
    elif kernel == "online-attention":
        arrays[3:] = [np.asarray(example_dimensions[name], dtype=np.int32) for name in ("m", "n")]
        scores = (arrays[0].astype(np.float64) @ arrays[1].astype(np.float64).T) * 0.25
        probability = np.exp(scores - scores.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        golden = (probability @ arrays[2].astype(np.float64)).astype(np.float32)
    elif kernel == "histogram":
        n, tiles = example_dimensions["n"], example_dimensions["tiles"]
        arrays[0] = rng.integers(-2, 10, size=n, dtype=np.int32)
        arrays[1:] = [np.asarray(value, dtype=np.int32) for value in (n, tiles)]
        golden = np.zeros((tiles, 8), dtype=np.int32)
        for program in range(tiles):
            lane = program * 16 + np.arange(16)
            selected = (lane < n) & (lane % 3 != 1)
            values = arrays[0][lane[selected]]
            golden[program] = np.bincount(values[(values >= 0) & (values < 8)], minlength=8).astype(np.int32)
    elif kernel == "runtime-memory-store":
        arrays[2] = np.asarray(3, dtype=np.int32)
        golden = arrays[1].copy()
        for _ in range(3):
            golden += arrays[0]
    elif kernel == "dynamic-count":
        arrays[1] = np.asarray(example_dimensions["n"], dtype=np.int32)
        golden = arrays[0] + np.float32((example_dimensions["n"] + 7) // 8)
    elif kernel == "dynamic-matmul":
        arrays[2:] = [np.asarray(example_dimensions[name], dtype=np.int32) for name in ("m", "n", "k")]
        golden = arrays[0] @ arrays[1]
    elif kernel == "masked-store":
        arrays[2] = np.asarray(example_dimensions["n"], dtype=np.int32)
        selected = (np.arange(example_dimensions["n"]) % 3 != 1) | (arrays[0] > 0)
        golden = arrays[1].copy()
        for update in range(2):
            golden[selected] = (arrays[0][selected] + np.float32(update)) + np.float32(1)
    elif kernel == "masked-matrix":
        arrays[2:] = [np.asarray(example_dimensions[name], dtype=np.int32) for name in ("m", "n")]
        r, c = np.indices(arrays[0].shape)
        selected = ((r + c) % 3 != 1) & (arrays[0] > 0)
        golden = arrays[1].T.copy()
        golden[selected] = arrays[0][selected] + np.float32(2)
        golden = golden.T.copy()
    elif kernel in ("indexed-gather", "dynamic-gather"):
        arrays[1] = np.array([0, 14, 7, -2147483648, 3, 3, -99, 8, 0, 14, 12, -1, 6], dtype=np.int32)
        if kernel == "dynamic-gather":
            arrays[2:] = [np.asarray(example_dimensions[name], dtype=np.int32) for name in ("m", "n", "count")]
        golden = np.full(13, -7, dtype=np.float32)
        active = arrays[1] >= 0
        golden[active] = arrays[0].ravel()[arrays[1][active]]
    elif kernel in ("matmul", "tiled-matmul", "loop-matmul", "ragged-matmul", "blockptr-matmul", "descriptor-matmul"):
        golden = arrays[0] @ arrays[1]
    elif kernel == "scan":
        golden = np.concatenate([np.cumsum(arrays[0][start:start + 16][::-1])[::-1] for start in range(0, 53, 16)])
    elif kernel == "tuple-scan":
        arrays = [rng.integers(0, 3, (2, 8)).astype("float32"), rng.integers(-2, 3, (2, 8)).astype("float32")]
        golden, second = np.empty_like(arrays[0]), np.empty_like(arrays[1])
        a = b = None
        for index in reversed(range(8)):
            a, b = ((arrays[0][:, index], arrays[1][:, index]) if a is None else
                    (arrays[0][:, index] * a, arrays[0][:, index] * b + arrays[1][:, index]))
            golden[:, index], second[:, index] = a, b
        additional = [second]
    else:
        e = np.exp(arrays[0] - arrays[0].max())
        golden = e / e.sum()
    write_shared_bundle(result, out, arrays, [golden, *additional], work=work / "native", python=python,
                        rtol=2e-5 if kernel == "online-attention" else 1e-5,
                        atol=6e-6 if kernel in ("loop-matmul", "ragged-matmul", "blockptr-matmul", "descriptor-matmul", "dynamic-matmul") else 2e-6)
    return {"bundle": str(out), "kernel": fn.__name__, "native_numerical_check": "passed",
            "profile": result.trace.get("profile", "shared-functional-tensor"), "merlin_staged_execution": "not-certified"}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="t2m", description="Provenance-preserving TTIR to tensor MLIR")
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("demo", help="capture, lower and check a real Triton kernel; no GPU needed")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--size", type=int, default=1024)
    d.add_argument("--block", type=int, default=256)
    d.add_argument("--operation", choices=("add", "multiply-add"), default="add")
    lower = commands.add_parser("lower", help="lower a saved TTIR/spec to standard MLIR plus provenance")
    lower.add_argument("ttir", type=Path)
    lower.add_argument("--spec", type=Path, required=True)
    lower.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("verify-bundle", help="check artifact hashes and ABI bindings")
    check.add_argument("bundle", type=Path)
    native = commands.add_parser("verify-mlir", help="strict verification with existing native torch-mlir bindings")
    native.add_argument("mlir", type=Path)
    native.add_argument("--python", type=Path, help="Python environment that already contains torch-mlir")
    native.add_argument("--lower-to-llvm", action="store_true", help="also check standard legalization to LLVM in the verifier process")
    shared = commands.add_parser("shared-lower", help="run an existing triton-shared optimizer and record its middle-layer boundary")
    shared.add_argument("ttir", type=Path)
    shared.add_argument("--out", type=Path, required=True)
    shared.add_argument("--optimizer", type=Path, help="existing triton-shared-opt executable")
    from .upstream import STAGES
    shared.add_argument("--stage", choices=tuple(STAGES), default="structured")
    shared.add_argument("--optimizer-sha256", help="optional exact executable-content pin")
    shared.add_argument("--timeout", type=float, default=30)
    tensor = commands.add_parser("shared-tensor", help="adapt shared tensor computation and structured memory into functional tensors")
    tensor.add_argument("ttir", type=Path)
    tensor.add_argument("--abi", type=Path, required=True)
    tensor.add_argument("--out", type=Path, required=True)
    tensor.add_argument("--optimizer", type=Path)
    tensor.add_argument("--memory-profile", choices=("tensor", "buffer", "guarded-buffer"), default="tensor")
    tensor.add_argument("--address-policy", choices=("structured", "source-words"), default="structured",
                        help="source-words requires guarded-buffer and retains source integer arithmetic")
    tensor.add_argument("--inputs", type=Path, help="optional NPZ with in0, in1, ...; requires --reference")
    tensor.add_argument("--reference", type=Path, help="optional independent NPZ with out0, out1, ...")
    tensor.add_argument("--python", type=Path, help="existing native torch-mlir environment, for bundle checks")
    tensor.add_argument("--rtol", type=float, default=0)
    tensor.add_argument("--atol", type=float, default=0)
    sd = commands.add_parser("shared-demo", help="package source-captured tensor kernels after native execution")
    sd.add_argument("--out", type=Path, required=True)
    sd.add_argument("--kernel", choices=("matrix-add", "softmax", "matmul", "tiled-add", "tiled-matmul", "loop-matmul", "ragged-matmul", "blockptr-matmul", "descriptor-matmul", "scan", "tuple-scan", "indexed-gather", "runtime-affine", "runtime-prefix", "runtime-recurrence", "runtime-memory-loop", "pointer-while", "selected-memory", "online-attention", "histogram", "runtime-memory-store", "dynamic-count", "dynamic-matmul", "dynamic-gather", "masked-store", "masked-matrix", "indirect-store"), default="softmax")
    sd.add_argument("--optimizer", type=Path)
    sd.add_argument("--memory-profile", choices=("tensor", "buffer", "guarded-buffer"), default="tensor")
    sd.add_argument("--address-policy", choices=("structured", "source-words"), default="structured")
    sd.add_argument("--python", type=Path, help="existing Python with torch-mlir")
    coverage = commands.add_parser("shared-coverage", help="measure a fixed upstream stage on a real MLIR fixture corpus")
    coverage.add_argument("corpus", type=Path)
    coverage.add_argument("--out", type=Path, required=True)
    coverage.add_argument("--optimizer", type=Path)
    coverage.add_argument("--stage", choices=tuple(STAGES), default="tensor-ir")
    coverage.add_argument("--limit", type=int, default=1000)
    coverage.add_argument("--budget", type=float, default=120)
    bind = commands.add_parser("bind-model", help="replace model2MLIR registered kernel calls and write a natively checked model bundle")
    bind.add_argument("capture", type=Path)
    bind.add_argument("--bindings", type=Path, required=True, help="t2m.kernel-bindings.v1 JSON with TTIR/ABI specializations")
    bind.add_argument("--out", type=Path, required=True)
    bind.add_argument("--optimizer", type=Path)
    bind.add_argument("--memory-profile", choices=("tensor", "buffer", "guarded-buffer"), default="tensor")
    bind.add_argument("--address-policy", choices=("structured", "source-words"), default="structured")
    bind.add_argument("--python", type=Path, help="existing native environment with torch-mlir and safetensors")
    bind.add_argument("--merlin-root", type=Path, help="optional existing Merlin checkout for real interface/runtime generation checks")
    bind.add_argument("--merlin-python", type=Path)
    bind.add_argument("--merlin-host", action="store_true", help="compile staged inputs with Merlin and execute its native host library")
    bind.add_argument("--merlin-clang", type=Path, help="explicit compatible native LLVM code generator")
    bind.add_argument("--rtol", type=float, default=0)
    bind.add_argument("--atol", type=float, default=0)
    qualify = commands.add_parser("qualify-merlin", help="stage and execute a standalone tensor bundle through Merlin")
    qualify.add_argument("capture", type=Path)
    qualify.add_argument("--out", type=Path, required=True)
    qualify.add_argument("--merlin-root", type=Path, required=True)
    qualify.add_argument("--merlin-python", type=Path)
    qualify.add_argument("--python", type=Path, help="existing native compiler environment with torch-mlir")
    qualify.add_argument("--merlin-clang", type=Path)
    qualify.add_argument("--rtol", type=float, default=0)
    qualify.add_argument("--atol", type=float, default=0)
    qualified = commands.add_parser("verify-merlin", help="verify standalone compiler/execution qualification artifacts")
    qualified.add_argument("qualification", type=Path)
    invoke = commands.add_parser("invoke-merlin", help="run a qualified native library with new arguments and references")
    invoke.add_argument("qualification", type=Path)
    invoke.add_argument("--inputs", type=Path, required=True)
    invoke.add_argument("--reference", type=Path, required=True, help="independent NPZ with out0, out1, ...")
    invoke.add_argument("--out", type=Path, required=True)
    invoke.add_argument("--merlin-root", type=Path, required=True)
    invoke.add_argument("--merlin-python", type=Path)
    invoke.add_argument("--rtol", type=float, default=0)
    invoke.add_argument("--atol", type=float, default=0)
    invocation = commands.add_parser("verify-merlin-invocation", help="verify reused-library invocation evidence")
    invocation.add_argument("invocation", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            summary = demo(args.out, size=args.size, block=args.block, operation=args.operation)
        elif args.command == "shared-demo":
            summary = shared_demo(args.out, kernel=args.kernel, optimizer=args.optimizer, python=args.python,
                                  memory_profile=args.memory_profile, address_policy=args.address_policy)
        elif args.command == "shared-coverage":
            from .coverage import qualify_corpus
            report = qualify_corpus(args.corpus, args.out, optimizer=args.optimizer, stage=args.stage,
                                    limit=args.limit, budget=args.budget)
            summary = {key: report[key] for key in ("measured_cases", "discovered_cases", "complete_corpus_measured",
                       "stage_successes", "source_native_verified", "inventory_failures", "elapsed_seconds")}
            summary["functional_tensor_coverage"] = report["functional_tensor_coverage"]
        elif args.command == "bind-model":
            from .model_binding import KernelBinding
            from .model_bundle import write_model_bundle
            from .shared_tensor import TensorABI, convert_shared
            configuration = json.loads(args.bindings.read_text())
            if configuration.get("schema") != "t2m.kernel-bindings.v1" or not isinstance(configuration.get("kernels"), list):
                raise ValueError("expected a t2m.kernel-bindings.v1 configuration")
            if args.out.exists():
                raise FileExistsError(args.out)
            work = args.out.with_name(args.out.name + ".work")
            work.mkdir(parents=True, exist_ok=False)
            bindings = []
            for i, item in enumerate(configuration["kernels"]):
                abi = item["abi"]
                if isinstance(abi, str):
                    abi = json.loads((args.bindings.parent / abi).read_text())
                implementation = convert_shared(args.bindings.parent / item["ttir"], TensorABI.from_dict(abi),
                                                work / f"kernel-{i}", optimizer=args.optimizer,
                                                memory_profile=item.get("memory_profile", args.memory_profile),
                                                address_policy=item.get("address_policy", args.address_policy))
                bindings.append(KernelBinding(item["kernel"], implementation, operands=item.get("operands", {}),
                                               results=tuple(item.get("results", ())), scalar_arguments=item.get("scalar_arguments", {})))
            write_model_bundle(args.capture, bindings, args.out, work=work / "model", python=args.python,
                               merlin_root=args.merlin_root, merlin_python=args.merlin_python,
                               merlin_host=args.merlin_host, merlin_clang=args.merlin_clang,
                               rtol=args.rtol, atol=args.atol)
            summary = {"bundle": str(args.out), "native_model_comparison": "passed",
                       "merlin_interface": "passed" if args.merlin_root else "not-tested",
                       "merlin_host_execution": "passed" if args.merlin_host else "not-performed",
                       "merlin_target_execution": "not-performed"}
        elif args.command == "qualify-merlin":
            from .kernel_consumer import qualify_kernel
            qualify_kernel(args.capture, args.out, merlin_root=args.merlin_root, python=args.merlin_python,
                           compiler_python=args.python, clang=args.merlin_clang, rtol=args.rtol, atol=args.atol)
            summary = {"qualification": str(args.out), "compiler_input_staging": "passed",
                       "merlin_native_host_execution": "passed", "accelerator_execution": "not-performed"}
        elif args.command == "invoke-merlin":
            from .invocation import invoke_kernel
            invoke_kernel(args.qualification, args.inputs, args.reference, args.out, merlin_root=args.merlin_root,
                          python=args.merlin_python, rtol=args.rtol, atol=args.atol)
            summary = {"invocation": str(args.out), "reference_comparison": "passed",
                       "native_compilation": "reused-qualified-library", "accelerator_execution": "not-performed"}
        elif args.command == "verify-merlin-invocation":
            from .invocation import verify_invocation
            report = verify_invocation(args.invocation)
            summary = {"invocation": str(args.invocation), "artifacts_verified": len(report["artifacts"])}
        elif args.command == "verify-merlin":
            from .kernel_consumer import verify_kernel_qualification
            receipt = verify_kernel_qualification(args.qualification)
            summary = {"qualification": str(args.qualification), "artifacts_verified": len(receipt["artifacts"])}
        elif args.command == "verify-bundle":
            receipt = verify_bundle(args.bundle)
            summary = {"bundle": str(args.bundle), "artifacts_verified": len(receipt["artifacts"])}
        elif args.command == "verify-mlir":
            from .verify import verify_mlir
            summary = verify_mlir(args.mlir.read_text(), python=args.python, lower_to_llvm=args.lower_to_llvm)
        elif args.command == "shared-lower":
            from .upstream import lower_with_shared
            report = lower_with_shared(args.ttir, args.out, optimizer=args.optimizer, stage=args.stage,
                                       timeout=args.timeout, expected_optimizer_sha256=args.optimizer_sha256)
            summary = {"out": str(args.out), "stage": report["stage"],
                       "stage_completed": True, "portable_conversion_complete": False,
                       "next_step": "qualify functional tensor ABI and provenance; see stage.json"}
        elif args.command == "shared-tensor":
            from .shared_tensor import TensorABI, convert_shared
            from .shared_bundle import write_shared_bundle
            if bool(args.inputs) != bool(args.reference):
                raise ValueError("provide both --inputs and --reference to publish a checked bundle")
            if args.out.exists():
                raise FileExistsError(args.out)
            work = args.out.with_name(args.out.name + ".work")
            abi = TensorABI.from_dict(json.loads(args.abi.read_text()))
            result = convert_shared(args.ttir, abi, work / "translation", optimizer=args.optimizer,
                                    memory_profile=args.memory_profile, address_policy=args.address_policy)
            if args.inputs:
                with np.load(args.inputs, allow_pickle=False) as saved:
                    if set(saved.files) != {f"in{i}" for i in range(len(abi.inputs))}:
                        raise ValueError("inputs must contain exactly in0, in1, ... in ABI order")
                    inputs = [saved[f"in{i}"].copy() for i in range(len(saved.files))]
                with np.load(args.reference, allow_pickle=False) as saved:
                    if set(saved.files) != {f"out{i}" for i in range(len(abi.outputs))}:
                        raise ValueError("reference must contain exactly out0, out1, ... in result order")
                    reference = [saved[f"out{i}"].copy() for i in range(len(saved.files))]
                write_shared_bundle(result, args.out, inputs, reference, work=work / "native", python=args.python,
                                    rtol=args.rtol, atol=args.atol)
            else:
                args.out.mkdir(parents=True, exist_ok=False)
                (args.out / "model.mlir").write_text(result.mlir_text)
                (args.out / "provenance.json").write_text(json.dumps(result.trace, indent=2, sort_keys=True) + "\n")
                (args.out / "spec.json").write_text(json.dumps(abi.as_dict(), indent=2, sort_keys=True) + "\n")
                for name, payload in result.auxiliary_artifacts.items():
                    (args.out / name).write_text(payload)
            summary = {"out": str(args.out), "profile": result.trace.get("profile", "shared-functional-tensor"), "checked_bundle": bool(args.inputs),
                       "merlin_staged_execution": "not-certified"}
        else:
            spec = KernelSpec.from_dict(json.loads(args.spec.read_text()))
            result = convert(args.ttir, spec)
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / "model.mlir").write_text(result.mlir_text)
            (args.out / "provenance.json").write_text(json.dumps(result.trace, indent=2, sort_keys=True) + "\n")
            summary = {"mlir": str(args.out / "model.mlir"), "bundle": False, "conversion": "complete"}
        print(json.dumps(summary, indent=2))
        return 0
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, f"t2m: {exc}\n")
