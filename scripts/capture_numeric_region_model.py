#!/usr/bin/env python3
"""Capture real models with numeric map callbacks and independent CPU references."""

import argparse
import sys
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2m.examples.numeric_region_kernels import numeric_region_values


def capture(out, dtype="f32"):
    out = Path(out).resolve()
    if dtype not in ("f32", "i64"):
        raise ValueError("the numeric model example requires f32 or i64 storage")
    storage = torch.float32 if dtype == "f32" else torch.int64
    name = "t2m_demo::numeric_regions"
    families = ("histogram", "gather", "dot", "scaled-dot", "dot-zero")

    @triton_op(name, mutates_args={})
    def numeric(x: torch.Tensor, y: torch.Tensor, initial: torch.Tensor,
                n: int, kind: int) -> torch.Tensor:
        result = torch.empty_like(x)
        if n:
            wrap_triton(numeric_region_values)[((n + 7) // 8,)](
                x, y, initial, result, n, B=8, FAMILY=families[kind])
        return result

    @numeric.register_kernel("cpu")
    def reference(x, y, initial, n, kind):
        # Separate source-level references, without frontend or lowering helpers.
        # Finite dyadic f32 inputs keep these contraction examples exact at each
        # source multiply/add; this does not assert a general GPU dot policy.
        if kind in (2, 3, 4):
            return ((32 if kind == 3 else 16) * x * y
                    + (0.0 if kind == 4 else initial)) * 256
        result = x.clone()
        mask = (1 << 64) - 1
        for index in range(n):
            word = int(x[index])
            if kind == 0:
                value = (word & mask).bit_count()
            else:
                value = sum((word + ((slot + (word & 7)) & 7)) * (slot + 1)
                            for slot in range(8)) & mask
                if value >= 1 << 63:
                    value -= 1 << 64
            result[index] = value
        return result

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if dtype == "f32":
                values = [.125, -.25, .5, -.5, .25]
                weight = torch.tensor([values[index % len(values)] for index in range(19)], dtype=storage)
                bias = (torch.arange(19, dtype=storage) % 5 - 2) * .125
            else:
                weight = torch.tensor([(-1 if index % 2 else 1) * (1 << 59) + index * 13
                                       for index in range(19)], dtype=storage)
                bias = torch.arange(19, dtype=storage) - 7
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.bias = torch.nn.Parameter(bias, requires_grad=False)

        def forward(self, x):
            original = x + self.weight
            if dtype == "f32":
                contracted = numeric(original, self.weight, self.bias, 19, 2)
                scaled = numeric(original, self.weight, self.bias, 19, 3)
                zero = numeric(original, self.weight, self.bias, 19, 4)
                dependent = numeric(contracted, self.weight, self.bias, 19, 2)
                empty = numeric(x[:0], x[:0], x[:0], 0, 2)
                return dependent, contracted, scaled, zero, empty, original, contracted + self.bias
            histogram = numeric(original, self.weight, self.bias, 19, 0)
            gathered = numeric(original, self.weight, self.bias, 19, 1)
            dependent = numeric(gathered, histogram, self.bias, 19, 0)
            empty = numeric(x[:0], x[:0], x[:0], 0, 0)
            return dependent, histogram, gathered, empty, original, gathered + self.bias

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        x = (torch.arange(19, dtype=storage) % 9 - 4) * .125 if dtype == "f32" else (
            torch.arange(19, dtype=storage) * 37 - (1 << 58))
        inputs = (x,)
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
    parser.add_argument("--dtype", choices=("f32", "i64"), default="f32")
    args = parser.parse_args()
    print(capture(args.out, args.dtype))
