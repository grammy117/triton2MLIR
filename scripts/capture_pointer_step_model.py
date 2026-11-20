#!/usr/bin/env python3
"""Capture dependent bidirectional pointer streams with live singleton controls."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, *, final_only=False):
    source = Path(__file__).resolve().parents[1] / "t2m/examples/runtime_header_kernels.py"
    loader = importlib.util.spec_from_file_location("pointer_step_source", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    kernel = kernels.final_pointer_mix if final_only else kernels.bidirectional_sum
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::pointer_step"

    @triton_op(name, mutates_args={})
    def transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, stride: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi, step = lower.item(), upper.item(), stride.item()
        if not (0 <= lo <= 3 and 0 <= hi <= 8 and 1 <= step <= 3):
            raise ValueError("pointer-step controls outside the source contract")
        result, last = torch.empty((17,), dtype=x.dtype, device=x.device), torch.empty((17,), dtype=x.dtype, device=x.device)
        options = {"MIX": True} if final_only else {}
        wrap_triton(kernel)[(3,)](x, result, last, lo, hi, step, N=17, R=16, B=8, TRANSPOSE=False, **options)
        return result, last

    @transform.register_fake
    def fake(x, lower, upper, stride):
        return (torch.empty((17,), dtype=x.dtype, device=x.device), torch.empty((17,), dtype=x.dtype, device=x.device))

    @transform.register_kernel("cpu")
    def reference(x, lower, upper, stride):
        lo, hi, step = lower.item(), upper.item(), stride.item()
        if not (0 <= lo <= 3 and 0 <= hi <= 8 and 1 <= step <= 3):
            raise ValueError("pointer-step controls outside the source contract")
        acc, final_row = torch.zeros((17,), dtype=x.dtype), lo
        state = x[0].clone()
        for row in range(lo, hi, step):
            if final_only:
                state = (state ^ (row + 2)) * 3
            else:
                acc = acc + (x[row] ^ (x[15-row] * 3))
            final_row = row + step
        if final_only:
            acc = state ^ x[final_row]
        return acc, x[final_row] - x[15-final_row] * 3

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(16*17, dtype=torch.int64).reshape(16, 17) * 13 + (1 << 54), requires_grad=False)

        def forward(self, x, lower, upper, stride):
            first, earlier = transform(x, lower, upper, stride)
            shifted = first + self.weight
            second, final = transform(shifted, lower, upper, stride)
            return second, first, earlier, final, x, shifted, self.weight, lower, upper, stride

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        model = Model().eval()
        x = torch.arange(16*17, dtype=torch.int64).reshape(16, 17) * 37 + (1 << 55)
        x[1::2] *= -1
        inputs = (x, torch.tensor(1, dtype=torch.int64), torch.tensor(8, dtype=torch.int64), torch.tensor(3, dtype=torch.int64))
        converted = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
            weights_path=str(out / "weights.safetensors"))
        if not converted.ok:
            raise ValueError(converted.diagnostics)
        write_bundle(model, inputs, out, conversion_result=converted, capture_trace=True,
            capture_regions=False, source_path=Path(__file__))
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(name, None)
        else:
            KERNEL_REGISTRY[name] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(capture(args.out, final_only=args.final_only))
