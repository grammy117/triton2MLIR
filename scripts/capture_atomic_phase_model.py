#!/usr/bin/env python3
"""Capture dependent synchronized atomic kernels with complete state snapshots."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, dtype="i64", *, volatile=False):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / "t2m/examples" / (
        "atomic_volatile_kernels.py" if volatile else "atomic_barrier_kernels.py")
    spec = importlib.util.spec_from_file_location("atomic_phase_model_kernel", source)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    kernel = kernels.atomic_volatile_updates if volatile else kernels.atomic_phases
    state_size = 32 if volatile else 16
    name = "t2m_demo::atomic_phases"

    @triton_op(name, mutates_args={})
    def exchange(x: torch.Tensor, state: torch.Tensor, n: int, steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        updated, old, seen = state.clone(), torch.empty_like(x), torch.empty_like(x)
        wrap_triton(kernel)[((n + 7) // 8,)](
            x, updated, old, seen, n, steps, B=8, CROSS=False)
        return updated, old, seen

    @exchange.register_kernel("cpu")
    def reference(x, state, n, steps):
        assert tuple(x.shape) == (n,) and tuple(state.shape) == (state_size,)
        updated = state.clone()
        old, seen = torch.full_like(x, -7), torch.full_like(x, -7)
        for _ in range(steps):
            for origin in range(0, n, 8):
                for lane in range(origin, min(origin + 8, n)):
                    slot = origin // 2 + (lane - origin) % 4
                    old[lane] = updated[slot].clone()
                    updated[slot] += x[lane]
                for lane in range(origin, min(origin + 8, n)):
                    seen[lane] = updated[origin // 2 + (lane - origin) % 4]
        return updated, old, seen

    dt = torch.int64 if dtype == "i64" else torch.float32

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(2, dtype=dt), requires_grad=False)
            self.state = torch.nn.Parameter(torch.arange(state_size, dtype=dt) - 3, requires_grad=False)

        def forward(self, x):
            original = x + self.bias
            first, old1, seen1 = exchange(original, self.state, x.shape[0], 2)
            second, old2, seen2 = exchange(original, first, x.shape[0], 3)
            empty, old0, seen0 = exchange(x[:0], second, 0, 1)
            return second, first, old1, seen1, old2, seen2, original, x, empty, old0, seen0, self.state, self.bias

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
    parser.add_argument("--volatile", action="store_true", help="retain private volatile observations alongside atomic phases")
    args = parser.parse_args()
    print(capture(args.out, args.dtype, volatile=args.volatile))
