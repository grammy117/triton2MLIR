#!/usr/bin/env python3
"""Capture dependent Triton calls that retain selected roots and while state."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, kind):
    source = Path(__file__).resolve().parents[1] / "t2m/examples/root_state_kernels.py"
    loader = importlib.util.spec_from_file_location("root_state_source", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    kernel = getattr(kernels, kind)
    name = "t2m_demo::root_state_" + kind
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)

    @triton_op(name, mutates_args={})
    def transform(x: torch.Tensor, y: torch.Tensor, lower: torch.Tensor,
                  upper: torch.Tensor, stride: torch.Tensor,
                  memory_stride: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi, step, advance = lower.item(), upper.item(), stride.item(), memory_stride.item()
        if not (0 <= lo <= 3 and 0 <= hi <= 8 and 1 <= step <= 3 and -2 <= advance <= 2):
            raise ValueError("root-state controls outside the source contract")
        first, last = (torch.empty((17,), dtype=x.dtype, device=x.device) for _ in range(2))
        options = {"PARTIAL": False} if kind == "different_root_choice" else {}
        wrap_triton(kernel)[(3,)](x, y, first, last, lo, hi, step, advance, N=17, B=8, **options)
        return first, last

    @transform.register_fake
    def fake(x, y, lower, upper, stride, memory_stride):
        return (torch.empty((17,), dtype=x.dtype, device=x.device),
                torch.empty((17,), dtype=x.dtype, device=x.device))

    @transform.register_kernel("cpu")
    def reference(x, y, lower, upper, stride, memory_stride):
        lo, hi, step, advance = lower.item(), upper.item(), stride.item(), memory_stride.item()
        if not (0 <= lo <= 3 and 0 <= hi <= 8 and 1 <= step <= 3 and -2 <= advance <= 2):
            raise ValueError("root-state controls outside the source contract")
        state, position, selected = x[0].clone(), 16 + lo, x
        for row in range(lo, hi, step):
            if kind == "different_root_choice":
                state = state + (x[16 + row*advance] if row < 3 else y[row-2])
            else:
                state = state + selected[position]
                if kind == "while_progress":
                    position += advance if row < 3 else 1
                elif row < 3:
                    position += advance
                else:
                    selected, position = y, row-2
        value = selected[position]
        # The source stores into two independently allocated result buffers.
        return (state.clone(), state.clone()) if kind == "different_root_choice" else (state ^ value, value.clone())

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(64*17, dtype=torch.int64).reshape(64, 17)*13 + (1 << 54),
                requires_grad=False)
            self.table = torch.nn.Parameter(
                -torch.arange(32*17, dtype=torch.int64).reshape(32, 17)*19 - (1 << 55),
                requires_grad=False)

        def forward(self, x, lower, upper, stride, memory_stride):
            first, earlier = transform(x, self.table, lower, upper, stride, memory_stride)
            shifted = first + self.weight
            second, final = transform(shifted, self.table, lower, upper, stride, memory_stride)
            return (second, first, earlier, final, x, shifted, self.weight, self.table,
                    lower, upper, stride, memory_stride, first[:0])

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        model = Model().eval()
        x = torch.arange(64*17, dtype=torch.int64).reshape(64, 17)*37 + (1 << 58)
        x[1::2] *= -1
        inputs = (x, *(torch.tensor(v, dtype=torch.int64) for v in (1, 8, 3, -2)))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kind", choices=("different_root_choice", "different_root_loop", "while_progress"), required=True)
    args = parser.parse_args()
    capture(args.out, args.kind)
