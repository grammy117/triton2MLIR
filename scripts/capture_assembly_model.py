#!/usr/bin/env python3
"""Capture learned integer state, dependent source-assembly calls and six results."""

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


def capture(out, *, dtype="i64"):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    root = Path(__file__).resolve().parents[1]
    kernels = load("assembly_model_kernels", root / "t2m/examples/assembly_kernels.py")
    sample = load("assembly_model_reference", root / "adapters/ptx/examples/model_reference.py")
    width = 64 if dtype == "i64" else 32
    scalar_type = torch.int64 if width == 64 else torch.int32
    assembly = sample.program(width)
    qualname = "t2m_demo::assembly_word_state"

    @triton_op(qualname, mutates_args={})
    def assembly_word_state(x: torch.Tensor, y: torch.Tensor, shifts: torch.Tensor,
                            n: int, iterations: int) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = torch.empty_like(x), torch.empty_like(y)
        wrap_triton(kernels.assembly_word_state)[((n + 31) // 32,)](
            x.contiguous().view(-1), y.contiguous().view(-1), shifts.contiguous().view(-1), a.view(-1), b.view(-1),
            n, iterations, ASM=assembly, B=32, WORDS64=width == 64)
        return a, b

    @assembly_word_state.register_kernel("cpu")
    def reference(x, y, shifts, n, iterations):
        if n != x.numel():
            raise ValueError("reference domain differs from captured word count")
        arrays = [v.detach().contiguous().numpy() for v in (x, y, shifts)]
        return tuple(torch.from_numpy(v) for v in sample.reference(*arrays, iterations))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            weight = torch.arange(32, dtype=scalar_type).reshape(4, 8) * 13 - 17
            if width == 64:
                weight += 1 << 60
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.other = torch.nn.Parameter((torch.arange(32, dtype=scalar_type).reshape(4, 8) - 19) * 7, requires_grad=False)
            self.register_buffer("shifts", torch.tensor([0, 1, width - 1, width, width + 1, -1, 3, 7] * 4,
                                                        dtype=torch.int32).reshape(4, 8))

        def forward(self, x):
            first = assembly_word_state(x + self.weight, self.other, self.shifts, 32, 3)
            second = assembly_word_state(first[0], first[1], self.shifts, 32, 2)
            empty = assembly_word_state(x[:0], self.other[:0], self.shifts[:0], 0, 4)
            return *first, *second, *empty

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (torch.arange(32, dtype=scalar_type).reshape(4, 8) * 5 - 11,)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(model, inputs, backend="fx_importer", weights_path=str(out / "weights.safetensors"), capture_trace=True)
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
    args = parser.parse_args()
    print(capture(args.out, dtype=args.dtype))
