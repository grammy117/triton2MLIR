#!/usr/bin/env python3
"""Capture dependent volatile state updates and complete original snapshots."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, dtype="i64"):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    path = Path(__file__).resolve().parents[1] / "t2m/examples/volatile_kernels.py"
    spec = importlib.util.spec_from_file_location("volatile_model_source", path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    name = "t2m_demo::volatile_updates"

    @triton_op(name, mutates_args={})
    def update(x: torch.Tensor, state: torch.Tensor, n: int, steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result, old, seen = state.clone(), torch.empty_like(x), torch.empty_like(x)
        wrap_triton(source.volatile_updates)[((n + 7) // 8,)](x, result, old, seen, n, steps, B=8)
        return result, old, seen

    @update.register_kernel("cpu")
    def reference(x, state, n, steps):
        assert tuple(x.shape) == tuple(state.shape) == (n,)
        result, old, seen = state.clone(), torch.full_like(x, -7), torch.full_like(x, -7)
        for _ in range(steps):
            old = result.clone()
            result.add_(x)
            seen = result.clone()
        return result, old, seen

    dt = torch.int64 if dtype == "i64" else torch.float32

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(2, dtype=dt), requires_grad=False)
            self.initial = torch.nn.Parameter(torch.tensor(-3, dtype=dt), requires_grad=False)

        def forward(self, x):
            original, initial = x + self.bias, x + self.initial
            first, old1, seen1 = update(original, initial, x.shape[0], 2)
            second, old2, seen2 = update(original, first, x.shape[0], 3)
            empty, old0, seen0 = update(x[:0], initial[:0], 0, 1)
            return second, first, old1, seen1, old2, seen2, original, x, initial, empty, old0, seen0, self.bias, self.initial

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (torch.arange(13, dtype=dt) - 6,)
        exported = torch.export.export(model, inputs,
            dynamic_shapes=({0: torch.export.Dim("length", min=2, max=31)},))
        result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True,
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
    parser.add_argument("--dtype", choices=("i64", "f32"), default="i64")
    args = parser.parse_args()
    print(capture(args.out, args.dtype))
