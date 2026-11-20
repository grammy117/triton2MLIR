#!/usr/bin/env python3
"""Capture dependent Triton while updates with observable condition memory."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out):
    source = Path(__file__).resolve().parents[1] / "t2m/examples/while_tensor_kernels.py"
    loader = importlib.util.spec_from_file_location("while_state_kernel", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    kernel = kernels.checked_while_update
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::while_state"

    @triton_op(name, mutates_args={"state", "checks"})
    def update(state: torch.Tensor, checks: torch.Tensor, limit: int, step: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        before, after, seen = (torch.empty_like(state) for _ in range(3))
        wrap_triton(kernel)[(1,)](state, checks, before, after, seen, limit, step)
        return before, after, seen

    @update.register_kernel("cpu")
    def reference(state, checks, limit, step):
        before = state.clone()
        while True:
            seen = checks.clone()
            checks.add_(1)
            if not bool(state[0] < limit):
                break
            state.add_(step)
        return before, state.clone(), seen

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.counter = torch.nn.Parameter(torch.tensor([(1 << 55) + 1], dtype=torch.int64), requires_grad=False)

        def forward(self, x):
            # A source-model bound, not a compiler-invented trip restriction.
            state, checks = x.clamp(0, 9).clone(), self.counter.clone()
            before = state.clone(), checks.clone()
            first = update(state, checks, 5, 2)
            middle = state.clone(), checks.clone()
            second = update(state, checks, 9, 3)
            return *before, *first, *middle, *second, state, checks, x, self.counter

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        model, x = Model().eval(), torch.tensor([1], dtype=torch.int64)
        exported = torch.export.export(model, (x,))
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
