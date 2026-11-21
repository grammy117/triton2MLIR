#!/usr/bin/env python3
"""Capture dependent Triton calls with independent signed memory strides."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, mode="memory", *, conditional=False):
    source = Path(__file__).resolve().parents[1] / "t2m/examples" / ("conditional_stride_kernels.py" if conditional else "runtime_stride_kernels.py")
    loader = importlib.util.spec_from_file_location("invariant_stride_source", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::conditional_stride" if conditional else "t2m_demo::invariant_stride"
    kernel = kernels.conditional_stride if conditional else kernels.invariant_stride

    @triton_op(name, mutates_args={})
    def transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
                  stride: torch.Tensor, memory_stride: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi, step, advance = lower.item(), upper.item(), stride.item(), memory_stride.item()
        if not (0 <= lo <= 3 and 0 <= hi <= 8 and 1 <= step <= 3 and -2 <= advance <= 2):
            raise ValueError("invariant-stride controls outside the source contract")
        result, last = (torch.empty((17,), dtype=x.dtype, device=x.device) for _ in range(2))
        wrap_triton(kernel)[(3,)](x, result, last, lo, hi, step, advance,
            N=17, R=64, B=8, TRANSPOSE=False, MODE=mode)
        return result, last

    @transform.register_fake
    def fake(x, lower, upper, stride, memory_stride):
        return (torch.empty((17,), dtype=x.dtype, device=x.device), torch.empty((17,), dtype=x.dtype, device=x.device))

    @transform.register_kernel("cpu")
    def reference(x, lower, upper, stride, memory_stride):
        lo, hi, step, advance = lower.item(), upper.item(), stride.item(), memory_stride.item()
        if not (0 <= lo <= 3 and 0 <= hi <= 8 and 1 <= step <= 3 and -2 <= advance <= 2):
            raise ValueError("invariant-stride controls outside the source contract")
        state, position = x[0].clone(), 16 + lo
        for row in range(lo, hi, step):
            if mode == "pure":
                state = (state ^ (row + 2)) * 3
            else:
                state = state + (x[position] ^ (x[63-position] * 3))
            if conditional:
                position += int(row < 3) if mode == "bit" else advance if row < 3 else -advance if mode == "nested" and row >= 5 else 1 // (row-2)
                if mode == "observe":
                    state ^= x[position]
            else:
                position += step + advance if mode == "mixed" else 1 // hi if mode == "late" else advance
        return state ^ x[position], x[position] - x[63-position] * 3

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(64*17, dtype=torch.int64).reshape(64, 17) * 13 + (1 << 54), requires_grad=False)

        def forward(self, x, lower, upper, stride, memory_stride):
            first, earlier = transform(x, lower, upper, stride, memory_stride)
            shifted = first + self.weight
            second, final = transform(shifted, lower, upper, stride, memory_stride)
            return second, first, earlier, final, x, shifted, self.weight, lower, upper, stride, memory_stride

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        model = Model().eval()
        x = torch.arange(64*17, dtype=torch.int64).reshape(64, 17) * 37 + (1 << 55)
        x[1::2] *= -1
        inputs = (x, *[torch.tensor(v, dtype=torch.int64) for v in (1, 8, 3, -2)])
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
    parser.add_argument("--mode", choices=("memory", "pure", "mixed", "late"), default="memory")
    parser.add_argument("--conditional-mode", choices=("scalar", "pointer", "pure", "observe", "nested", "bit"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(capture(args.out, args.conditional_mode or args.mode, conditional=bool(args.conditional_mode)))
