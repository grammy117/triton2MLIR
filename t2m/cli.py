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


def shared_demo(out, *, kernel="softmax", optimizer=None, python=None):
    from .examples import tensor_kernels as kernels
    from .frontend import capture_ttir
    from .shared_tensor import TensorABI, TensorBuffer, convert_shared
    from .shared_bundle import write_shared_bundle
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    work.mkdir(parents=True, exist_ok=False)
    grid = (1,)
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
    elif kernel == "matrix-add":
        fn, constants, shapes = kernels.matrix_add, {"M": 4, "N": 8}, [(4, 8)] * 3
    elif kernel == "matmul":
        fn, constants, shapes = kernels.block_matmul, {"M": 16, "N": 16, "K": 16}, [(16, 16)] * 3
    elif kernel == "scan":
        fn, constants, shapes, grid = kernels.scan_tiles, {"N": 53, "B": 16, "KIND": 0, "REVERSE": True}, [(53,)] * 2, (4,)
    elif kernel == "tuple-scan":
        fn, constants, shapes = kernels.tuple_scan, {"R": 2, "S": 8, "AXIS": 1, "REVERSE": True}, [(2, 8)] * 4
    else:
        fn, constants, shapes = kernels.row_softmax, {"N": 7, "BLOCK": 8}, [(7,)] * 2
    names = ["x", "y", "out"] if len(shapes) == 3 else ["x", "out"]
    if kernel == "tuple-scan":
        names = ["a", "b", "out_a", "out_b"]
    output_effect = "write" if kernel in ("ragged-matmul", "blockptr-matmul", "descriptor-matmul", "scan", "tuple-scan") else "readwrite"
    abi = TensorABI(fn.__name__, tuple(TensorBuffer(n, s, output_effect if n.startswith("out") else "read")
                    for n, s in zip(names, shapes, strict=True)), grid=grid, intent=f"Source {kernel} example")
    source = work / "kernel.ttir"
    source.write_text(capture_ttir(fn, {**{n: "*fp32" for n in names},
                                      **{n: "constexpr" for n in constants}}, constants))
    result = convert_shared(source, abi, work / "translation", optimizer=optimizer)
    rng = np.random.default_rng(7)
    arrays = [rng.normal(size=b.shape).astype("float32") for b in abi.inputs]
    additional = []
    if kernel in ("matrix-add", "tiled-add"):
        golden = arrays[0] + arrays[1]
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
                        rtol=1e-5, atol=6e-6 if kernel in ("loop-matmul", "ragged-matmul", "blockptr-matmul", "descriptor-matmul") else 2e-6)
    return {"bundle": str(out), "kernel": fn.__name__, "native_numerical_check": "passed",
            "profile": "shared-functional-tensor", "merlin_staged_execution": "not-certified"}


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
    tensor.add_argument("--inputs", type=Path, help="optional NPZ with in0, in1, ...; requires --reference")
    tensor.add_argument("--reference", type=Path, help="optional independent NPZ with out0, out1, ...")
    tensor.add_argument("--python", type=Path, help="existing native torch-mlir environment, for bundle checks")
    tensor.add_argument("--rtol", type=float, default=0)
    tensor.add_argument("--atol", type=float, default=0)
    sd = commands.add_parser("shared-demo", help="package source-captured tensor kernels after native execution")
    sd.add_argument("--out", type=Path, required=True)
    sd.add_argument("--kernel", choices=("matrix-add", "softmax", "matmul", "tiled-add", "tiled-matmul", "loop-matmul", "ragged-matmul", "blockptr-matmul", "descriptor-matmul", "scan", "tuple-scan"), default="softmax")
    sd.add_argument("--optimizer", type=Path)
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
    bind.add_argument("--python", type=Path, help="existing native environment with torch-mlir and safetensors")
    bind.add_argument("--merlin-root", type=Path, help="optional existing Merlin checkout for real interface/runtime generation checks")
    bind.add_argument("--merlin-python", type=Path)
    bind.add_argument("--merlin-host", action="store_true", help="compile staged inputs with Merlin and execute its native host library")
    bind.add_argument("--merlin-clang", type=Path, help="explicit compatible native LLVM code generator")
    bind.add_argument("--rtol", type=float, default=0)
    bind.add_argument("--atol", type=float, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            summary = demo(args.out, size=args.size, block=args.block, operation=args.operation)
        elif args.command == "shared-demo":
            summary = shared_demo(args.out, kernel=args.kernel, optimizer=args.optimizer, python=args.python)
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
                                                work / f"kernel-{i}", optimizer=args.optimizer)
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
            result = convert_shared(args.ttir, abi, work / "translation", optimizer=args.optimizer)
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
            summary = {"out": str(args.out), "profile": "shared-functional-tensor", "checked_bundle": bool(args.inputs),
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
