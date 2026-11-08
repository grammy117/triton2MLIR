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
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            summary = demo(args.out, size=args.size, block=args.block, operation=args.operation)
        elif args.command == "verify-bundle":
            receipt = verify_bundle(args.bundle)
            summary = {"bundle": str(args.bundle), "artifacts_verified": len(receipt["artifacts"])}
        elif args.command == "verify-mlir":
            from .verify import verify_mlir
            summary = verify_mlir(args.mlir.read_text(), python=args.python, lower_to_llvm=args.lower_to_llvm)
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
