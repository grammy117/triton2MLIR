#!/usr/bin/env python3
"""Capture two selected-pointer Triton calls with independent PyTorch goldens."""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.library import triton_op, wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


def capture(out, *, dtype="f32"):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = Path(__file__).resolve().parents[1] / "t2m/examples/selected_pointer_kernels.py"
    spec = importlib.util.spec_from_file_location("t2m_selected_kernels", source)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::selected_gather"
    tensor_dtype = {"f32": torch.float32, "f16": torch.float16}[dtype]

    @triton_op(qualname, mutates_args={})
    def selected_gather(x: torch.Tensor, y: torch.Tensor, choice: int = 0) -> torch.Tensor:
        out = torch.empty((13,), dtype=x.dtype, device=x.device)
        wrap_triton(kernels.selected_permute)[(1,)](x, y, out, n=13, m=29, choice=choice, B=8, MODE=2)
        return out

    @selected_gather.register_kernel("cpu")
    def reference(x, y, choice=0):
        selected = (torch.arange(13) * 6 + choice) % 16
        return torch.where(selected < 8, x[selected % 8], y[28 - selected % 8])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(29, dtype=tensor_dtype))

        def forward(self, x):
            return selected_gather(x, self.weight, choice=0) * 0.25 + selected_gather(x, self.weight, choice=1)

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(53)
        model, values = Model().eval(), (torch.randn(29, dtype=tensor_dtype),)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(model, values, backend="fx_importer",
                             weights_path=str(out / "weights.safetensors"), capture_trace=True)
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, values, out, capture_regions=False, capture_trace=True,
                     source_path=Path(__file__), conversion_result=result)
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(qualname, None)
        else:
            KERNEL_REGISTRY[qualname] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    args = parser.parse_args()
    print(capture(args.out, dtype=args.dtype))
