#!/usr/bin/env python3
"""Capture dependent interleaved tiles and all intermediate observations."""

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
    source = Path(__file__).resolve().parents[1] / "t2m/examples/interleaved_tile_kernels.py"
    loader = importlib.util.spec_from_file_location("interleaved_source", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::interleaved_partition"

    @triton_op(name, mutates_args={})
    def transform(x: torch.Tensor, bias: torch.Tensor, n: int, control: int, kind: int
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, sa, sb = (torch.empty_like(x) for _ in range(4))
        wrap_triton(kernels.interleaved_partition)[(triton.cdiv(n, 8),)](
            x, bias, a, b, sa, sb, n, control, B=8, MODE="cross" if kind == 0 else "readonly")
        return a, b, sa, sb

    @transform.register_kernel("cpu")
    def reference(x, bias, n, control, kind):
        assert x.numel() == n
        value, p = x ^ bias[0], x > 0
        q = p ^ (control == 0)
        first_a = value ^ 1
        early_a = torch.where(p, first_a, -17)
        first_b = early_a ^ value ^ 2 if kind == 0 else value ^ 2
        early_b = torch.where(q, first_b, -31)
        second_a = early_b ^ value ^ 4 if kind == 0 else value ^ 4
        second_b = early_a ^ value ^ 8 if kind == 0 else value ^ 8
        a, b = torch.where(p, first_a, second_a), torch.where(q, first_b, second_b)
        return a, b, early_a if kind == 0 else a.clone(), early_b if kind == 0 else b.clone()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([(1 << 54) + 1], dtype=torch.int64), requires_grad=False)

        def forward(self, x):
            n = x.shape[0]
            a, b, sa, sb = transform(x, self.bias, n, n - 2, 0)
            shifted = a + b
            c, d, sc, sd = transform(shifted, self.bias, n, n - 2, 1)
            return c, d, a, b, sa, sb, sc, sd, x, shifted, self.bias

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
