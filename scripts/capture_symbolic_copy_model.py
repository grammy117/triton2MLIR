#!/usr/bin/env python3
"""Capture a real exported model with two dependent dynamic Triton copies."""

import argparse
import sys
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2m.examples.live_scalar_kernels import live_extent_copy, live_extent_weighted


def capture(out, *, weighted=False):
    name = "t2m_demo::symbolic_weighted" if weighted else "t2m_demo::symbolic_copy"

    if weighted:
        @triton_op(name, mutates_args={})
        def copy(x: torch.Tensor, weight: torch.Tensor, n: int) -> torch.Tensor:
            out = torch.empty_like(x)
            wrap_triton(live_extent_weighted)[((n + 7) // 8,)](x, weight, n, out, B=8)
            return out

        @copy.register_kernel("cpu")
        def reference(x, weight, n):
            assert x.numel() == n
            return x ^ weight[0]

        @copy.register_fake
        def fake(x, weight, n):
            return torch.empty_like(x)
    else:
        @triton_op(name, mutates_args={})
        def copy(x: torch.Tensor, n: int) -> torch.Tensor:
            out = torch.empty_like(x)
            wrap_triton(live_extent_copy)[((n + 7) // 8,)](x, n, out, B=8)
            return out

        @copy.register_kernel("cpu")
        def reference(x, n):
            assert x.numel() == n
            return x.clone()

        @copy.register_fake
        def fake(x, n):
            return torch.empty_like(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if weighted:
                self.weight = torch.nn.Parameter(torch.tensor([17], dtype=torch.int32), requires_grad=False)

        def forward(self, x):
            if weighted:
                first = copy(x, self.weight, x.shape[0])
                second = copy(first, self.weight, first.shape[0])
                return second, first, x, self.weight
            first = copy(x, x.shape[0])
            second = copy(first, first.shape[0])
            return second, first, x

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (torch.arange(13, dtype=torch.int32),)
        program = torch.export.export(model, inputs,
            dynamic_shapes={"x": {0: torch.export.Dim("n", min=0, max=31)}})
        out = Path(out).resolve()
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(program, inputs, backend="fx_importer", capture_trace=True,
                             weights_path=str(out / "weights.safetensors"))
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, inputs, out, capture_trace=True, capture_regions=False,
                     conversion_result=result, source_path=Path(__file__))
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(name, None)
        else:
            KERNEL_REGISTRY[name] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weighted", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, weighted=args.weighted))
