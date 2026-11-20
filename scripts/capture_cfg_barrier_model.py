#!/usr/bin/env python3
"""Capture dependent early-exit tile phases with complete state snapshots."""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.library import triton_op, wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


def capture(out, dtype="i64", effect="ordinary"):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / "t2m/examples/cfg_barrier_kernels.py"
    spec = importlib.util.spec_from_file_location("cfg_barrier_model_kernel", source)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    atomic, volatile = effect in ("typed-atomics", "atomic-volatile"), effect in ("volatile", "atomic-volatile")
    name = "t2m_demo::cfg_program_exchange"

    @triton_op(name, mutates_args={})
    def exchange(state: torch.Tensor, old: torch.Tensor, seen: torch.Tensor, n: int, steps: int, mode: int
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        updated, previous, observed = state.clone(), old.clone(), seen.clone()
        wrap_triton(kernels.early_tile_exchange)[((n + 3) // 4,)](
            updated, previous, observed, n, steps, mode, B=4, ATOMIC=atomic, VOLATILE=volatile)
        return updated, previous, observed

    @exchange.register_kernel("cpu")
    def reference(state, old, seen, n, steps, mode):
        assert tuple(state.shape) == tuple(old.shape) == tuple(seen.shape) == (n, 4)
        updated, previous, observed = state.clone(), old.clone(), seen.clone()
        for origin in range(0, n, 4):
            if (origin // 4 + mode) % 2 == 0:
                continue
            size = min(4, n - origin)
            for iteration in range(steps):
                previous[origin:origin + size] = updated[origin:origin + size]
                updated[origin:origin + size] += iteration + 1
                tile = torch.full((4, 4), -3, dtype=state.dtype, device=state.device)
                tile[:size] = updated[origin:origin + size]
                observed[origin:origin + size] = tile.T[:size]
        return updated, previous, observed

    dt = torch.int64 if dtype == "i64" else torch.float32

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([1, -2, 3, -4], dtype=dt), requires_grad=False)

        def forward(self, x):
            original = x + self.bias
            old, seen = torch.full_like(original, -17), torch.full_like(original, -23)
            first, a, b = exchange(original, old, seen, x.shape[0], 2, 0)
            second, c, d = exchange(first, a, b, x.shape[0], 3, 1)
            empty, e, f = exchange(x[:0], a[:0], b[:0], 0, 1, 1)
            return second, first, a, b, c, d, original, x, empty, e, f, self.bias

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (torch.arange(28, dtype=dt).reshape(7, 4) - 6,)
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
    parser.add_argument("--dtype", choices=("i64", "f32"), default="i64")
    parser.add_argument("--effect-profile", choices=("ordinary", "typed-atomics", "volatile", "atomic-volatile"),
                        default="ordinary")
    args = parser.parse_args()
    print(capture(args.out, args.dtype, args.effect_profile))
