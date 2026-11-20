#!/usr/bin/env python3
"""Capture dependent partial-tile recurrences with three live source loop controls."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
import triton
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out):
    source = Path(__file__).resolve().parents[1] / "t2m/examples/runtime_header_kernels.py"
    loader = importlib.util.spec_from_file_location("runtime_header_source", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::runtime_partition"

    @triton_op(name, mutates_args={})
    def transform(x: torch.Tensor, bias: torch.Tensor, n: int, lower: int, upper: int, stride: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        result, seen = torch.empty_like(x), torch.empty_like(x)
        wrap_triton(kernels.runtime_partition)[(triton.cdiv(n, 8),)](x, bias, result, seen, n, lower, upper, stride, B=8, MISSING_ZERO=False)
        return result, seen

    @transform.register_kernel("cpu")
    def reference(x, bias, n, lower, upper, stride):
        assert x.numel() == n
        word = x ^ bias[0]
        p, state, snapshot = word > 0, word ^ 1, word ^ 64
        for iteration in range(lower, upper, stride):
            earlier = torch.where(p, state, -17)
            state = torch.where(p, state, (word ^ (iteration + 2)) ^ earlier)
            snapshot = snapshot ^ state
            state = torch.where(p, state ^ 8, state)
        if lower >= upper:
            state = torch.where(p, state, word ^ 16)
        return state, snapshot

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([(1 << 54) + 1], dtype=torch.int64), requires_grad=False)

        def forward(self, x):
            n = x.shape[0]
            first, earlier = transform(x, self.bias, n, n // 4, n - 2, n // 8 + 1)
            shifted = first + self.bias
            second, final = transform(shifted, self.bias, n, n // 8, n, n // 4 + 1)
            return second, first, earlier, final, x, shifted, self.bias

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        model, x = Model().eval(), torch.arange(17, dtype=torch.int64) + (1 << 55)
        x[1::2] *= -1
        exported = torch.export.export(model, (x,), dynamic_shapes=({0: torch.export.Dim("n", min=2, max=31)},))
        converted = m2m.convert(exported, (x,), backend="fx_importer", capture_trace=True,
                                weights_path=str(out / "weights.safetensors"))
        if not converted.ok:
            raise ValueError(converted.diagnostics)
        write_bundle(model, (x,), out, conversion_result=converted, capture_trace=True,
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
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(capture(args.out))
