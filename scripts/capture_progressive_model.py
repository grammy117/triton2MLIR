#!/usr/bin/env python3
"""Capture dependent Triton calls observing each completed row or column."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out):
    source = Path(__file__).resolve().parents[1] / "t2m/examples/partial_memory_kernels.py"
    loader = importlib.util.spec_from_file_location("progressive_kernel", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    kernel = kernels.progressive_transform
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::progressive_transform"

    @triton_op(name, mutates_args={})
    def transform(x: torch.Tensor, bias: torch.Tensor, rows: int, cols: int, order: int
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = torch.empty_like(x)
        first = torch.empty((rows if order == 0 else cols,), dtype=x.dtype, device=x.device)
        last = torch.empty_like(first)
        wrap_triton(kernel)[(1,)](x, bias, result, first, last, rows, cols, ORDER=order, MODE="reread")
        return result, first, last

    @transform.register_kernel("cpu")
    def reference(x, bias, rows, cols, order):
        assert tuple(x.shape) == (rows, cols)
        result = x ^ bias.reshape((rows, 1) if order == 0 else (1, cols))
        planes = result if order == 0 else result.T
        checksum = torch.zeros((planes.shape[0],), dtype=x.dtype)
        for index in range(planes.shape[1]):
            checksum ^= planes[:, index]
        last = planes[:, -1].clone() if planes.shape[1] else torch.full_like(checksum, -23)
        return result, checksum, last

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.row_bias = torch.nn.Parameter(torch.arange(7, dtype=torch.int64) + (1 << 54), requires_grad=False)
            self.col_bias = torch.nn.Parameter(torch.arange(9, dtype=torch.int64) * 19 + (1 << 53) + 1, requires_grad=False)

        def forward(self, x):
            rows, cols = x.shape
            first, a, b = transform(x, self.row_bias[:rows].clone(), rows, cols, 0)
            second, c, d = transform(first, self.col_bias[:cols].clone(), rows, cols, 1)
            return second, first, a, b, c, d, x, self.row_bias, self.col_bias

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        model, x = Model().eval(), torch.arange(24, dtype=torch.int64).reshape(4, 6) + (1 << 55)
        exported = torch.export.export(model, (x,), dynamic_shapes=({0: torch.export.Dim("rows", min=2, max=7),
                                                                   1: torch.export.Dim("cols", min=2, max=9)},))
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
