#!/usr/bin/env python3
"""Capture dependent tile branches, observations and conditional division."""

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
    source = Path(__file__).resolve().parents[1] / "t2m/examples/branch_tile_kernels.py"
    loader = importlib.util.spec_from_file_location("branch_source", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::branch_partition"

    @triton_op(name, mutates_args={})
    def transform(x: torch.Tensor, bias: torch.Tensor, n: int, control: int, kind: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        result, seen = torch.empty_like(x), torch.empty_like(x)
        wrap_triton(kernels.branch_partition)[(triton.cdiv(n, 8),)](x, bias, result, seen, n, control, B=8,
            MODE="branch-observed" if kind == 0 else "guarded")
        return result, seen

    @transform.register_kernel("cpu")
    def reference(x, bias, n, control, kind):
        assert x.numel() == n
        word = x ^ bias[0]
        p, first = word > 0, word ^ 1
        if kind == 0:
            out = torch.where(p, first, word ^ (2 if control != 0 else 4))
            return out, torch.where(p, first, -17 if control != 0 else -31)
        negative = torch.div(word & 65535, control, rounding_mode="trunc") if control > 0 else word ^ 4
        out = torch.where(p, first, negative)
        return out, out.clone()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([(1 << 54) + 1], dtype=torch.int64), requires_grad=False)

        def forward(self, x):
            n = x.shape[0]
            first, earlier = transform(x, self.bias, n, n - 2, 0)
            shifted = first + self.bias
            second, final = transform(shifted, self.bias, n, n - 2, 1)
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
