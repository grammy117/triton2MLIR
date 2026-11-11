#!/usr/bin/env python3
"""Capture learned packed scalar control, input trip counts and all four results."""

import argparse
import importlib.util
from pathlib import Path
import sys

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, *, dtype="i64", pack=4, while_loop=True):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    examples = Path(__file__).resolve().parents[1] / "t2m/examples"
    sys.path.insert(0, str(examples))
    spec = importlib.util.spec_from_file_location("packed_control_capture_sources", examples / "packed_control_kernels.py")
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::packed_control"
    scalar_type = {"i64": torch.int64, "f32": torch.float32}[dtype]

    @triton_op(qualname, mutates_args={})
    def implementation(x: torch.Tensor, y: torch.Tensor, counts: torch.Tensor,
                       n: int, iterations: int) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = torch.empty_like(x), torch.empty_like(x)
        wrap_triton(kernels.packed_control)[((n + 31) // 32,)](
            x, y, counts, first, second, n, iterations, B=32, PACK=pack, WHILE=while_loop)
        return first, second

    @implementation.register_kernel("cpu")
    def reference(x, y, counts, n, iterations):
        divisor = torch.where(y != 0, y, torch.ones_like(y))
        quotient = x / divisor if x.dtype.is_floating_point else torch.div(x, divisor, rounding_mode="trunc")
        value = torch.where(y != 0, quotient, torch.full_like(x, -7))
        trips = counts + iterations
        for step in range(max(0, int(trips.max().item())) if trips.numel() else 0):
            value = torch.where(trips > step, value + y * step, value)
        return value, x + y

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            values = torch.tensor([0, -2, 4, 1, -1, 0, 2], dtype=scalar_type).repeat(3)[:17]
            self.weight = torch.nn.Parameter(values, requires_grad=False)

        def forward(self, x, counts):
            first = implementation(x, self.weight, counts, 17, 2)
            second = implementation(first[0], self.weight, counts, 17, 0)
            return *first, *second

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        x = torch.arange(17, dtype=scalar_type) * 3 - 17
        if dtype == "i64":
            x += 1 << 60
        counts = torch.tensor([0, 3, -4, 1, 2], dtype=torch.int32).repeat(4)[:17]
        model, values = Model().eval(), (x, counts)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(model, values, backend="fx_importer", weights_path=str(out / "weights.safetensors"), capture_trace=True)
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, values, out, capture_regions=False, capture_trace=True,
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
    parser.add_argument("--dtype", choices=("i64", "f32"), default="i64")
    parser.add_argument("--pack", type=int, choices=(2, 4), default=4)
    parser.add_argument("--for-loop", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, dtype=args.dtype, pack=args.pack, while_loop=not args.for_loop))
