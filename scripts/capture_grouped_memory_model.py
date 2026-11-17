#!/usr/bin/env python3
"""Capture explicit source-grouped memory, learned state and fifteen results."""

import argparse
import importlib.util
import sys
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


def capture(out, *, dtype="i64", warps=4, while_loop=True, execution_sizes=False):
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

    compact_names = ("t2m_grouped_demo::positive_count", "t2m_grouped_demo::compact") if execution_sizes else ()
    if execution_sizes:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from t2m.examples.kernel_size_kernels import positive_count, positive_pack

        @triton_op(compact_names[0], mutates_args={})
        def count_positive(x: torch.Tensor, capacity: int) -> torch.Tensor:
            count = torch.empty((1,), dtype=torch.int64, device=x.device)
            wrap_triton(positive_count)[(1,)](x, count, capacity, B=512)
            return count

        @count_positive.register_kernel("cpu")
        def count_reference(x, capacity):
            assert x.numel() == capacity
            return (x > 0).to(torch.int64).sum().reshape(1)

        @triton_op(compact_names[1], mutates_args={})
        def compact(x: torch.Tensor, n: int, capacity: int) -> torch.Tensor:
            result = torch.empty((n,), dtype=x.dtype, device=x.device)
            wrap_triton(positive_pack)[(1,)](x, result, capacity, n, B=512)
            return result

        @compact.register_kernel("cpu")
        def compact_reference(x, n, capacity):
            assert x.numel() == capacity
            return x[x > 0][:n].clone()

        @compact.register_fake
        def compact_fake(x, n, capacity):
            return torch.empty((n,), dtype=x.dtype, device=x.device)

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
            results = (*first, *second, *empty)
            if execution_sizes:
                flat = second[0].reshape(-1)
                count = count_positive(flat, 320)
                n = count.item()
                torch._check_is_size(n)
                torch._check(n <= 320)
                return *results, compact(flat, n, 320), count
            return results

    previous = {name: KERNEL_REGISTRY.get(name) for name in (qualname, *compact_names)}
    for name in previous:
        KERNEL_REGISTRY[name] = KernelSpec(name)
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
        for name, value in previous.items():
            if value is None:
                KERNEL_REGISTRY.pop(name, None)
            else:
                KERNEL_REGISTRY[name] = value
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("i64", "f32"), default="i64")
    parser.add_argument("--warps", type=int, choices=(2, 4), default=4)
    parser.add_argument("--for-loop", action="store_true")
    parser.add_argument("--execution-sizes", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, dtype=args.dtype, warps=args.warps, while_loop=not args.for_loop,
                  execution_sizes=args.execution_sizes))
