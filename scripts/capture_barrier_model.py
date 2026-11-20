#!/usr/bin/env python3
"""Capture dependent program-barrier kernels with snapshots and an empty call."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / "t2m/examples/barrier_kernels.py"
    spec = importlib.util.spec_from_file_location("program_barrier_model_kernel", source)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    name = "t2m_demo::program_exchange"

    @triton_op(name, mutates_args={})
    def exchange(x: torch.Tensor, n: int, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        state, observed = x.clone(), torch.empty_like(x)
        wrap_triton(kernels.exchange_tile)[((n + 3) // 4,)](
            state, observed, n, steps, B=4, CROSS=False
        )
        return state, observed

    @exchange.register_kernel("cpu")
    def reference(x, n, steps):
        assert tuple(x.shape) == (n, 4)
        state, observed = x.clone(), torch.full_like(x, -7)
        for iteration in range(steps):
            state.add_(iteration + 1)
            for origin in range(0, n, 4):
                size = min(4, n - origin)
                tile = torch.full((4, 4), -3, dtype=x.dtype, device=x.device)
                tile[:size] = state[origin : origin + size]
                observed[origin : origin + size] = tile.T[:size]
        return state, observed

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([1, -2, 3, -4], dtype=torch.int64), requires_grad=False)

        def forward(self, x):
            original = x + self.bias
            first, a = exchange(original, x.shape[0], 2)
            second, b = exchange(first, x.shape[0], 3)
            empty, z = exchange(x[:0], 0, 4)
            return second, first, a, b, original, x, empty, z, self.bias

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (torch.arange(28, dtype=torch.int64).reshape(7, 4) - 6,)
        program = torch.export.export(model, inputs,
            dynamic_shapes=({0: torch.export.Dim("rows", min=2, max=19)},))
        result = m2m.convert(program, inputs, backend="fx_importer", capture_trace=True,
                             weights_path=str(out / "weights.safetensors"))
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, inputs, out, conversion_result=result, capture_trace=True,
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
    print(capture(args.out))
