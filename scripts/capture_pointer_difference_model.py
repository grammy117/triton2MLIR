#!/usr/bin/env python3
"""Capture relative-pointer models with saved state and independent CPU math."""

import argparse
import sys
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2m.examples.pointer_address_kernels import relative_words


def capture(out, width=64):
    out = Path(out).resolve()
    if width not in (32, 64):
        raise ValueError("the relative-pointer model requires i32 or i64 words")
    storage = torch.int32 if width == 32 else torch.int64
    name = "t2m_demo::pointer_difference"

    @triton_op(name, mutates_args={})
    def relative(x: torch.Tensor, n: int, origin: int, stride: int,
                 capacity: int, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.empty((n,), dtype=x.dtype, device=x.device)
        distances = torch.empty((n,), dtype=torch.int64, device=x.device)
        if n:
            wrap_triton(relative_words)[((n + 7) // 8,)](
                x, values, distances, n, origin, stride, capacity, steps, B=8, WIDTH=width)
        return values, distances

    @relative.register_kernel("cpu")
    def reference(x, n, origin, stride, capacity, steps):
        assert x.numel() == capacity
        values = torch.full((n,), -17, dtype=x.dtype)
        distances = torch.zeros((n,), dtype=torch.int64)
        # Source-level logical coordinates; no compiler pointer/root helper or
        # physical address supplies the reference or complete model goldens.
        for step in range(steps):
            for lane in range(n):
                offset = (lane * stride + step) * (1 if lane % 2 == 0 else -1)
                values[lane] = x[origin + offset]
                distances[lane] = offset * (width // 8)
        return values, distances

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(256, dtype=storage) % 13 - 6, requires_grad=False)
            self.bias = torch.nn.Parameter(torch.arange(19, dtype=storage) * 17 - 91, requires_grad=False)

        def forward(self, x):
            original = x + self.weight
            values, distance = relative(original, 19, 128, 3, 256, 3)
            follow, follow_distance = relative(values, 7, 9, 1, 19, 2)
            empty, empty_distance = relative(x[:0], 0, 256, -8, 0, 4)
            zero, zero_distance = relative(original, 1, 256, -8, 256, 0)
            return (follow, follow_distance, values, distance, empty, empty_distance,
                    zero, zero_distance, original, values + self.bias)

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (torch.arange(256, dtype=storage) * 7919 + (1 << (29 if width == 32 else 58)),)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
                             weights_path=str(out / "weights.safetensors"))
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, inputs, out, capture_regions=False, capture_trace=True,
                     source_path=Path(__file__), conversion_result=result)
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(name, None)
        else:
            KERNEL_REGISTRY[name] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", choices=(32, 64), type=int, default=64)
    args = parser.parse_args()
    print(capture(args.out, args.width))
