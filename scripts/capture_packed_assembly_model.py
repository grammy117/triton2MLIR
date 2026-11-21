#!/usr/bin/env python3
"""Capture actual packed kernels, dependent calls, learned words and empty results."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(out, *, dtype="i64", pack=4):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    root = Path(__file__).resolve().parents[1]
    kernels = load("packed_assembly_kernels", root / "t2m/examples/packed_assembly_kernels.py")
    sample = load("packed_assembly_reference", root / "t2m/examples/packed_assembly_reference.py")
    width = 64 if dtype == "i64" else 32
    scalar_type = torch.int64 if width == 64 else torch.int32
    assembly, constraints = sample.program(width, pack)
    qualname = "t2m_demo::packed_assembly_state"

    @triton_op(qualname, mutates_args={})
    def packed_assembly_state(x: torch.Tensor, y: torch.Tensor,
                              n: int, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = torch.empty_like(x), torch.empty_like(y)
        wrap_triton(kernels.packed_assembly_state)[((n + 7) // 8,)](
            x.contiguous().view(-1), y.contiguous().view(-1), a.view(-1), b.view(-1),
            n, steps, ASM=assembly, CONSTRAINTS=constraints, PACK=pack, B=8,
            WORDS64=width == 64)
        return a, b

    @packed_assembly_state.register_kernel("cpu")
    def reference(x, y, n, steps):
        if n != x.numel() or x.shape != y.shape:
            raise ValueError("reference domain differs from captured word tensors")
        a, b = sample.reference(x.detach().contiguous().numpy(), y.detach().contiguous().numpy(), steps)
        return torch.from_numpy(a), torch.from_numpy(b)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            weight = torch.arange(32, dtype=scalar_type).reshape(4, 8) * 13 - 17
            if width == 64:
                weight += 1 << 60
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.other = torch.nn.Parameter((torch.arange(32, dtype=scalar_type).reshape(4, 8) - 19) * 7,
                                            requires_grad=False)

        def forward(self, x):
            first = packed_assembly_state(x + self.weight, self.other, 32, 3)
            second = packed_assembly_state(first[0], first[1], 32, 2)
            empty = packed_assembly_state(x[:0], self.other[:0], 0, 3)
            return x, *first, *second, *empty, self.weight, self.other

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (torch.arange(32, dtype=scalar_type).reshape(4, 8) * 5 - 11,)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(model, inputs, backend="fx_importer", weights_path=str(out / "weights.safetensors"),
                             capture_trace=True)
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, inputs, out, capture_regions=False, capture_trace=True,
                     source_path=Path(__file__), conversion_result=result)
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(qualname, None)
        else:
            KERNEL_REGISTRY[qualname] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("i64", "i32"), default="i64")
    parser.add_argument("--pack", type=int, choices=(2, 4, 16), default=4)
    args = parser.parse_args()
    print(capture(args.out, dtype=args.dtype, pack=args.pack))
