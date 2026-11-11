#!/usr/bin/env python3
"""Capture real tensor-model calls using external descriptor allocation views."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton
from triton.tools.tensor_descriptor import TensorDescriptor


def capture(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = (
        Path(__file__).resolve().parents[1] / "t2m/examples/descriptor_state_kernels.py"
    )
    spec = importlib.util.spec_from_file_location("descriptor_model_kernels", source)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::descriptor_state"

    @triton_op(qualname, mutates_args={})
    def descriptor_state(
        state: torch.Tensor,
        m: int,
        n: int,
        steps: int,
        capacity: int,
        read_origin: int,
        write_origin: int,
        read_row: int,
        read_col: int,
        write_row: int,
        write_col: int,
    ) -> torch.Tensor:
        updated = state.contiguous().clone()
        if m and n and steps:
            flat = updated.view(-1)
            first = TensorDescriptor(
                flat[read_origin:], [m, n], [read_row, read_col], [2, 16]
            )
            second = TensorDescriptor(
                flat[write_origin:], [m, n], [write_row, write_col], [2, 16]
            )
            wrap_triton(kernels.descriptor_state_updates)[
                ((m + 1) // 2, (n + 15) // 16)
            ](first, second, m, n, steps, capacity)
        return updated

    @descriptor_state.register_kernel("cpu")
    def reference(
        state,
        m,
        n,
        steps,
        capacity,
        read_origin,
        write_origin,
        read_row,
        read_col,
        write_row,
        write_col,
    ):
        if state.numel() != capacity:
            raise ValueError("descriptor model value differs from source capacity")
        updated = state.contiguous().clone()
        flat = updated.view(-1)
        for row in range(0, m, 2):
            for col in range(0, n, 16):
                coordinates = [
                    (i, j)
                    for i in range(row, min(m, row + 2))
                    for j in range(col, min(n, col + 16))
                ]
                reads = [
                    read_origin + i * read_row + j * read_col for i, j in coordinates
                ]
                writes = [
                    write_origin + i * write_row + j * write_col for i, j in coordinates
                ]
                for _ in range(steps):
                    snapshot = flat[reads].clone()
                    flat[writes] = snapshot + 1
        return updated

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(256, dtype=torch.int64).reshape(16, 16) + 2**53 + 1,
                requires_grad=False,
            )
            self.bias = torch.nn.Parameter(
                torch.arange(256, dtype=torch.int64).reshape(16, 16) % 7 - 3,
                requires_grad=False,
            )

        def forward(self, x):
            first = descriptor_state(
                x + self.weight, 2, 13, 2, 256, 8, 16, 24, 1, 24, 1
            )
            second = descriptor_state(first, 5, 19, 3, 256, 8, 8, 24, 1, 24, 1)
            empty = descriptor_state(x[:0], 0, 13, 2, 0, 8191, 8191, 24, 1, 24, 1)
            return second + self.bias, first, empty

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        values = (torch.arange(256, dtype=torch.int64).reshape(16, 16) % 5,)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(
            model,
            values,
            backend="fx_importer",
            weights_path=str(out / "weights.safetensors"),
            capture_trace=True,
        )
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(
            model,
            values,
            out,
            capture_regions=False,
            capture_trace=True,
            source_path=Path(__file__),
            conversion_result=result,
        )
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(qualname, None)
        else:
            KERNEL_REGISTRY[qualname] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(capture(parser.parse_args().out))
