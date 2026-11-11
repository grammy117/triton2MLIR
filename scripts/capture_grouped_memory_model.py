#!/usr/bin/env python3
"""Capture explicit source-grouped memory, learned state and fifteen results."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(out, *, dtype="i64", warps=4, while_loop=True):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    examples = Path(__file__).resolve().parents[1] / "t2m/examples"
    kernels = load("grouped_memory_model_sources", examples / "grouped_memory_kernels.py")
    oracle = load("grouped_memory_model_reference", examples / "grouped_memory_reference.py")
    qualname = "t2m_demo::grouped_read_state"
    scalar_type = {"i64": torch.int64, "f32": torch.float32}[dtype]

    @triton_op(qualname, mutates_args={})
    def implementation(x: torch.Tensor, y: torch.Tensor, n: int, capacity: int,
                       iterations: int, delta: int, source_target: str, source_warps: int,
                       source_threads: int, source_ctas: int, while_loop: bool,
                       selected: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_target != "cuda:80" or source_threads != 32 or source_ctas != 1:
            raise ValueError("unqualified example source specialization")
        state = x.contiguous().clone()
        outputs = [torch.empty((n,), dtype=x.dtype, device=x.device) for _ in range(4)]
        wrap_triton(kernels.grouped_read_state)[((n + 255) // 256,)](
            state, y, *outputs, n, capacity, iterations, delta,
            B=256, WHILE=while_loop, SELECT=selected, num_warps=source_warps)
        return state, *outputs

    @implementation.register_kernel("cpu")
    def reference(x, y, n, capacity, iterations, delta, source_target, source_warps,
                  source_threads, source_ctas, while_loop, selected):
        if source_target != "cuda:80" or source_threads != 32 or source_ctas != 1:
            raise ValueError("unqualified example source specialization")
        values = oracle.reference(x.detach().contiguous().numpy(), y.detach().contiguous().numpy(),
            n, capacity, iterations, delta, warps=source_warps, selected=selected)
        return tuple(torch.from_numpy(value) for value in values)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            values = torch.arange(320, dtype=scalar_type).reshape(10, 32) * -2 + 11
            if dtype == "i64":
                values -= 1 << 55
            self.weight = torch.nn.Parameter(values, requires_grad=False)

        def forward(self, x):
            profile = ("cuda:80", warps, 32, 1, while_loop, True)
            first = implementation(x, self.weight, 300, 320, 3, 2, *profile)
            second = implementation(first[0], self.weight, 319, 320, 1, -3, *profile)
            empty = implementation(x[:0], self.weight[:0], 0, 0, 2, 5, *profile)
            # The first complete state and all snapshots remain live after
            # the dependent call; destructive reuse must preserve their values.
            return *first, *second, *empty

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        x = torch.arange(320, dtype=scalar_type).reshape(10, 32) * 3 - 7
        if dtype == "i64":
            x += 1 << 56
        model, values = Model().eval(), (x,)
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
    parser.add_argument("--dtype", choices=("i64", "f32"), default="i64")
    parser.add_argument("--warps", type=int, choices=(2, 4), default=4)
    parser.add_argument("--for-loop", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, dtype=args.dtype, warps=args.warps, while_loop=not args.for_loop))
