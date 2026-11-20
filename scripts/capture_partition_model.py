#!/usr/bin/env python3
"""Capture dependent masked kernels with live control and learned word state."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
import triton
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, *, tiled=False):
    source = Path(__file__).resolve().parents[1] / "t2m/examples" / (
        "tile_partition_kernels.py" if tiled else "partition_kernels.py")
    loader = importlib.util.spec_from_file_location("partition_kernel", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    kernel = kernels.tile_partition if tiled else kernels.partition_transform
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::partition_transform"

    @triton_op(name, mutates_args={})
    def transform(x: torch.Tensor, bias: torch.Tensor, n: int, control: int, kind: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        result, seen = torch.empty_like(x), torch.empty_like(x)
        if tiled:
            wrap_triton(kernel)[(triton.cdiv(n, 8),)](x, bias, result, seen, n, control, B=8,
                MODE="nested" if kind == 0 else "observed")
        else:
            wrap_triton(kernel)[(1,)](x, bias, result, seen, n, control, MODE="nested" if kind == 0 else "observed")
        return result, seen

    @transform.register_kernel("cpu")
    def reference(x, bias, n, control, kind):
        assert x.numel() == n
        value = x ^ bias[0]
        positive = (x if tiled else value) > 0
        if kind == 0:
            result = torch.where(positive & (control != 0), value ^ 1, torch.where(positive, value ^ 2, value ^ 4))
            return result, result.clone()
        return torch.where(positive, value ^ 1, value ^ 2), torch.where(positive, value ^ 1, -17)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([(1 << 54) + 1], dtype=torch.int64), requires_grad=False)

        def forward(self, x):
            n = x.shape[0]
            first, a = transform(x, self.bias, n, n - 2, 0)
            shifted = first + self.bias
            second, b = transform(shifted, self.bias, n, n - 2, 1)
            return second, first, a, b, x, shifted, self.bias

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
    parser.add_argument("--tiled", action="store_true", help="Use parallel eight-lane tile partitions")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(capture(args.out, tiled=args.tiled))
