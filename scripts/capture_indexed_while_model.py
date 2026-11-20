#!/usr/bin/env python3
"""Capture dependent indexed while kernels with dynamic input and saved state."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out):
    source = Path(__file__).resolve().parents[1] / "t2m/examples/while_index_kernels.py"
    loader = importlib.util.spec_from_file_location("indexed_while_kernel", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    kernel = kernels.prefix_while
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::indexed_while"

    @triton_op(name, mutates_args={"state"})
    def update(x: torch.Tensor, state: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        before, after = torch.empty_like(state), torch.empty_like(state)
        wrap_triton(kernel)[(1,)](x, state, before, after, n)
        return before, after

    @update.register_kernel("cpu")
    def reference(x, state, n):
        assert n == x.numel()
        before = state.clone()
        for index in range(n):
            state.bitwise_xor_(x[index] + index)
        return before, state.clone()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.counter = torch.nn.Parameter(torch.tensor([(1 << 56) + 1], dtype=torch.int64), requires_grad=False)
            self.bias = torch.nn.Parameter(torch.tensor(3, dtype=torch.int64), requires_grad=False)

        def forward(self, x):
            state = self.counter.clone()
            before = state.clone()
            first = update(x, state, x.numel())
            middle = state.clone()
            shifted = x + self.bias
            second = update(shifted, state, shifted.numel())
            return before, *first, middle, *second, state, x, self.counter, self.bias, shifted

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        model, x = Model().eval(), torch.arange(17, dtype=torch.int64) + (1 << 55)
        exported = torch.export.export(model, (x,), dynamic_shapes=({0: torch.export.Dim("length", min=2, max=31)},))
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
