#!/usr/bin/env python3
"""Capture learned mapped reads, intervening state and all fourteen results."""

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


def capture(out, *, dtype="i64"):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    root = Path(__file__).resolve().parents[1]
    kernels = load("memory_map_model_kernels", root / "t2m/examples/elementwise_map_kernels.py")
    oracle = load("memory_map_model_reference", root / "t2m/examples/memory_map_reference.py")
    qualname = "t2m_demo::memory_map_state"
    scalar_type = torch.int64 if dtype == "i64" else torch.float32

    @triton_op(qualname, mutates_args={})
    def memory_map_state(x: torch.Tensor, n: int, capacity: int, iterations: int,
                         delta: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = x.contiguous().clone()
        outputs = [torch.empty((n,), dtype=x.dtype, device=x.device) for _ in range(4)]
        wrap_triton(kernels.memory_map_state)[(1,)](state.view(-1), *outputs, n, capacity, iterations, delta,
                                                    B=32, WHILE=True)
        return state, *outputs

    @memory_map_state.register_kernel("cpu")
    def reference(x, n, capacity, iterations, delta):
        results = oracle.reference(x.detach().contiguous().numpy(), n, capacity, iterations, delta)
        return tuple(torch.from_numpy(value) for value in results)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            weight = torch.arange(32, dtype=scalar_type).reshape(4, 8) * 3 - 7
            if dtype == "i64":
                weight += 1 << 60
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def forward(self, x):
            first = memory_map_state(x + self.weight, 13, 32, 3, 2)
            second = memory_map_state(first[0], 31, 32, 4, -3)
            empty = memory_map_state(x[:0], 0, 0, 4, 5)
            return second[0], *first[1:], *second[1:], *empty

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        values = (torch.arange(32, dtype=scalar_type).reshape(4, 8) * 5 - 11,)
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
    args = parser.parse_args()
    print(capture(args.out, dtype=args.dtype))
