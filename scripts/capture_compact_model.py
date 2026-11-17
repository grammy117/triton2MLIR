#!/usr/bin/env python3
"""Capture real count/scan/scatter compaction with a dependent Triton consumer."""

import argparse
import sys
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2m.examples.data_size_kernels import data_size_prefix
from t2m.examples.kernel_size_kernels import positive_count, positive_pack


def capture(out, *, matrix_views=False):
    count_name, pack_name, prefix_name = (
        "t2m_compact_demo::positive_count",
        "t2m_compact_demo::compact",
        "t2m_compact_demo::prefix",
    )

    @triton_op(count_name, mutates_args={})
    def count_positive(x: torch.Tensor, capacity: int) -> torch.Tensor:
        count = torch.empty((1,), dtype=torch.int64, device=x.device)
        wrap_triton(positive_count)[(1,)](x, count, capacity, B=32)
        return count

    @count_positive.register_kernel("cpu")
    def count_reference(x, capacity):
        assert x.numel() == capacity
        return (x > 0).to(torch.int64).sum().reshape(1)

    @triton_op(pack_name, mutates_args={})
    def compact(x: torch.Tensor, n: int, capacity: int) -> torch.Tensor:
        out = torch.empty((1, n) if matrix_views else (n,), dtype=x.dtype, device=x.device)
        wrap_triton(positive_pack)[(1,)](x, out, capacity, n, B=32)
        return out

    @compact.register_kernel("cpu")
    def compact_reference(x, n, capacity):
        assert x.numel() == capacity
        selected = x[x > 0]
        assert n <= selected.numel()
        result = selected[:n].clone()
        return result.reshape(1, n) if matrix_views else result

    @compact.register_fake
    def fake_compact(x, n, capacity):
        return torch.empty((1, n) if matrix_views else (n,), dtype=x.dtype, device=x.device)

    @triton_op(prefix_name, mutates_args={})
    def prefix(x: torch.Tensor, n: int, capacity: int) -> torch.Tensor:
        out = torch.empty((1, n) if matrix_views else (n,), dtype=x.dtype, device=x.device)
        wrap_triton(data_size_prefix)[((n + 7) // 8,)](x, n, capacity, out, B=8)
        return out

    @prefix.register_kernel("cpu")
    def prefix_reference(x, n, capacity):
        assert x.numel() == capacity
        result = torch.zeros((n,), dtype=x.dtype)
        result[: min(n, capacity)] = x.reshape(-1)[: min(n, capacity)]
        if matrix_views:
            result = result.reshape(1, n)
        return result

    @prefix.register_fake
    def fake_prefix(x, n, capacity):
        return torch.empty((1, n) if matrix_views else (n,), dtype=x.dtype, device=x.device)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor(-3, dtype=torch.int32), requires_grad=False
            )

        def forward(self, x):
            original = x + self.weight
            capacity = x.numel() if matrix_views else x.shape[0]
            count = count_positive(original, capacity)
            n = count.item()
            torch._check_is_size(n)
            torch._check(n <= 31)
            first = compact(original, n, capacity)
            second = prefix(first, n, n)
            tail = first[:, ::2] if matrix_views else first[::2]
            return second, first, original, x, count, self.weight, tail

    previous = {
        name: KERNEL_REGISTRY.get(name) for name in (count_name, pack_name, prefix_name)
    }
    try:
        for name in previous:
            KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (
            torch.tensor([4, -7, 2, 9, 1, 3, -2, 5, 0, 8, -1, 6, 7], dtype=torch.int32),
        )
        if matrix_views:
            inputs = (torch.tensor([[4, -7, 2, 9, 1], [3, -2, 5, 0, 8], [-1, 6, 7, -4, 2]], dtype=torch.int32),)
        program = torch.export.export(
            model,
            inputs,
            dynamic_shapes={"x": ({0: torch.export.Dim("rows", min=2, max=4),
                1: torch.export.Dim("cols", min=2, max=7)} if matrix_views else
                {0: torch.export.Dim("capacity", min=2, max=31)})},
        )
        out = Path(out).resolve()
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(
            program,
            inputs,
            backend="fx_importer",
            capture_trace=True,
            weights_path=str(out / "weights.safetensors"),
        )
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(
            model,
            inputs,
            out,
            capture_trace=True,
            capture_regions=False,
            conversion_result=result,
            source_path=Path(__file__),
        )
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
    parser.add_argument("--matrix-views", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, matrix_views=args.matrix_views))
