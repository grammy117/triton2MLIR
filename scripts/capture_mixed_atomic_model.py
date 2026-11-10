#!/usr/bin/env python3
"""Capture selected word atomics with floating and integer learned state."""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.library import triton_op, wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


def capture(out, width=32, kind="cas"):
    if kind not in ("cas", "rmw") or width not in ((16, 32, 64) if kind == "cas" else (32, 64)):
        raise ValueError("mixed atomic model requires CAS16/32/64 or RMW32/64")
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = Path(__file__).resolve().parents[1] / "t2m/examples/atomic_kernels.py"
    spec = importlib.util.spec_from_file_location("t2m_mixed_atomic_kernels", source)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    integer, floating = getattr(torch, f"int{width}"), getattr(torch, f"float{width}")
    capacity = 17 if kind == "cas" else 37
    qualname = "t2m_demo::mixed_atomic_update"

    @triton_op(qualname, mutates_args={})
    def mixed_atomic_update(x: torch.Tensor, index: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first, second = a.clone(), b.clone()
        old = torch.empty_like(x, dtype=floating)
        if kind == "cas":
            wrap_triton(kernels.selected_word_cas)[(2,)](first, second, index, x, old,
                choice=False, B=8, WIDTH=width, SCALAR=False, LANE=True)
        else:
            wrap_triton(kernels.selected_word_rmw)[(2,)](first, second, x, index, old,
                n=16, choice=True, B=8, WIDTH=width, SCALAR=False, OP="xor")
        return first, second, old

    @mixed_atomic_update.register_kernel("cpu")
    def reference(x, index, a, b):
        first, second = a.clone(), b.clone()
        old = torch.zeros_like(x, dtype=floating)
        words, previous = first.view(integer), old.view(integer)
        for lane in range(x.numel()):
            target = words if lane % 2 == 0 else second
            if kind == "cas":
                previous[lane] = target[lane]
                if target[lane] == index[lane]:
                    target[lane] = x[lane]
            elif lane % 3 != 1:
                location = int(index[lane])
                previous[lane] = target[location]
                target[location] ^= x[lane]
        return first, second, old

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.arange(capacity, dtype=floating) + 50, requires_grad=False)
            self.b = torch.nn.Parameter(torch.arange(capacity, dtype=integer) - 19, requires_grad=False)
            lanes = torch.arange(16)
            if kind == "cas":
                index = torch.where(lanes % 2 == 0, self.a.detach().view(integer)[:16], self.b.detach()[:16]).clone()
                index[1::3] ^= 1
            else:
                index = (lanes.to(torch.int32) * 11 + 3) % capacity
                index[lanes % 3 == 1] = -1000
            self.register_buffer("index", index)

        def forward(self, x):
            return mixed_atomic_update(x, self.index, self.a, self.b)

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model, values = Model().eval(), (torch.arange(16, dtype=integer) - 5,)
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
    parser.add_argument("--kind", choices=("cas", "rmw"), default="cas")
    args = parser.parse_args()
    print(capture(args.out, args.width, args.kind))
