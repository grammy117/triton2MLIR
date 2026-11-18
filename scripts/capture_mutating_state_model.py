#!/usr/bin/env python3
"""Capture actual mutating Triton calls with explicit functional state ports."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
import triton
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, kind="public"):
    source = Path(__file__).resolve().parents[1] / "t2m/examples/mutating_state.py"
    loader = importlib.util.spec_from_file_location("mutating_state_kernel", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    observe_and_bump = kernels.observe_and_bump
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::mutating_state"

    @triton_op(name, mutates_args={"state"})
    def update(state: torch.Tensor, n: int, bias: int) -> torch.Tensor:
        observed = torch.empty_like(state)
        wrap_triton(observe_and_bump)[(triton.cdiv(n, 8),)](state, observed, n, bias, B=8)
        return observed

    @update.register_kernel("cpu")
    def reference(state, n, bias):
        assert n == state.numel()
        observed = state.clone()
        state.add_(bias)
        return observed

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor((1 << 53) + 1, dtype=torch.int64), requires_grad=False)
            self.register_buffer("state", torch.arange(17, dtype=torch.int64) + 10)

        def forward(self, x):
            state = self.state if kind == "buffer" else x if kind == "public" else x + self.weight
            before = state.clone()
            first = update(state, state.numel(), 3)
            second = update(state, state.numel(), -2)
            return before, first, second, state, x, self.weight

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        model, x = Model().eval(), torch.arange(17, dtype=torch.int64)
        exported = torch.export.export(model, (x,), dynamic_shapes=None if kind == "buffer" else
            ({0: torch.export.Dim("length", min=2, max=31)},))
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
    parser.add_argument("--kind", choices=("internal", "public", "buffer"), default="public")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(capture(args.out, args.kind))
