#!/usr/bin/env python3
"""Capture integer/floating selected storage with actual model weights."""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.library import triton_op, wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


def capture(out, width=32):
    if width not in (16, 32, 64):
        raise ValueError("mixed view model requires width 16, 32 or 64")
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = Path(__file__).resolve().parents[1] / "t2m/examples/pointer_view_kernels.py"
    spec = importlib.util.spec_from_file_location("t2m_mixed_view_kernels", source)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    integer, floating = getattr(torch, f"int{width}"), getattr(torch, f"float{width}")
    qualname = "t2m_demo::mixed_word_view"

    @triton_op(qualname, mutates_args={})
    def mixed_word_view(x: torch.Tensor, state: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        out = initial.clone()
        wrap_triton(kernels.word_view_memory)[(2,)](x, state, out, n=13, steps=2,
            B=8, WIDTH=width, SELECT=True, BEFORE=True, WHILE=True, SCALAR=False)
        return out

    @mixed_word_view.register_kernel("cpu")
    def reference(x, state, initial):
        result = initial.clone()
        words = result.view(integer)
        choose = torch.arange(13) % 2 == 0
        active = torch.arange(13) % 3 != 1
        for row in range(2):
            selected = torch.where(choose, x[row], state.view(integer)[row])
            observed = torch.where(active, selected, torch.full_like(selected, -17))
            words[row] = observed ^ (-(1 << (width - 1)))
        return result

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state = torch.nn.Parameter(torch.arange(52, dtype=floating).reshape(4, 13) + 50, requires_grad=False)
            self.initial = torch.nn.Parameter(torch.arange(52, dtype=floating).reshape(4, 13) - 100, requires_grad=False)

        def forward(self, x):
            return mixed_word_view(x, self.state, self.initial)

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model, values = Model().eval(), (torch.arange(52, dtype=integer).reshape(4, 13) - 26,)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(model, values, backend="fx_importer", weights_path=str(out / "weights.safetensors"), capture_trace=True)
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
    parser.add_argument("--width", type=int, choices=(16, 32, 64), default=32)
    args = parser.parse_args()
    print(capture(args.out, args.width))
