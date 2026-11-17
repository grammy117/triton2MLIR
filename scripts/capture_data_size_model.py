#!/usr/bin/env python3
"""Capture an actual input word controlling two Triton result allocations."""

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


def capture(out):
    name = "t2m_demo::data_size_prefix"

    @triton_op(name, mutates_args={})
    def prefix(x: torch.Tensor, n: int, capacity: int) -> torch.Tensor:
        out = torch.empty((n,), dtype=x.dtype, device=x.device)
        wrap_triton(data_size_prefix)[((n + 7) // 8,)](x, n, capacity, out, B=8)
        return out

    @prefix.register_kernel("cpu")
    def reference(x, n, capacity):
        assert x.numel() == capacity
        out = torch.zeros((n,), dtype=x.dtype)
        used = min(n, capacity)
        out[:used] = x[:used]
        return out

    @prefix.register_fake
    def fake(x, n, capacity):
        return torch.empty((n,), dtype=x.dtype, device=x.device)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor(17, dtype=torch.int32), requires_grad=False
            )

        def forward(self, x, count):
            n = count.item()
            torch._check_is_size(n)
            torch._check(n <= 31)
            original = x + self.weight
            first = prefix(original, n, x.shape[0])
            second = prefix(first, n, n)
            return second, first, original, x, count, self.weight, first[::2]

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (
            torch.arange(13, dtype=torch.int32),
            torch.tensor(7, dtype=torch.int64),
        )
        program = torch.export.export(
            model,
            inputs,
            dynamic_shapes={
                "x": {0: torch.export.Dim("capacity", min=2, max=31)},
                "count": {},
            },
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
        if previous is None:
            KERNEL_REGISTRY.pop(name, None)
        else:
            KERNEL_REGISTRY[name] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(capture(parser.parse_args().out))
