#!/usr/bin/env python3
"""Capture complete model calls under the explicit mixed-FP8 source profile."""

import argparse
import importlib.util
import sys
from pathlib import Path

import m2m
import numpy as np
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton
from triton.tools.tensor_descriptor import TensorDescriptor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t2m.examples.fp8_allocation_reference import update


def capture(out):
    out = Path(out).resolve()
    source = ROOT / "t2m/examples/fp8_allocation_kernels.py"
    spec = importlib.util.spec_from_file_location("fp8_model_kernels", source)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::fp8_allocation"

    @triton_op(qualname, mutates_args={})
    def fp8_allocation(
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
        result = state.contiguous().clone()
        if m and n and steps:
            flat = result.view(-1)
            first = TensorDescriptor(
                flat[read_origin:], [m, n], [read_row, read_col], [2, 16]
            )
            second = TensorDescriptor(
                flat[write_origin:], [m, n], [write_row, write_col], [2, 16]
            )
            wrap_triton(kernels.descriptor_fp8_updates)[((m + 1) // 2, (n + 15) // 16)](
                first, second, m, n, steps, capacity
            )
        return result

    @fp8_allocation.register_kernel("cpu")
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
        # Independent rational FP8 palettes define the selected software source
        # contract. This is not an unspecified GPU backend-equivalence claim.
        return torch.from_numpy(
            update(
                state.detach().numpy(),
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
            )
        )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            palette = np.asarray(
                [1.0625, -1.125, 448, -448, 57344, 61440, 65536, 1e-20], np.float32
            )
            self.weight = torch.nn.Parameter(
                torch.from_numpy(np.resize(palette, 256).reshape(16, 16)),
                requires_grad=False,
            )
            self.bias = torch.nn.Parameter(
                torch.arange(256, dtype=torch.float32).reshape(16, 16) % 7 - 3,
                requires_grad=False,
            )

        def forward(self, x):
            original = x + self.weight
            first = fp8_allocation(original, 2, 13, 2, 256, 8, 16, 24, 1, 24, 1)
            second = fp8_allocation(first, 5, 19, 3, 256, 8, 8, 24, 1, 24, 1)
            empty = fp8_allocation(x[:0], 0, 13, 2, 0, 8191, 8191, 24, 1, 24, 1)
            return second + self.bias, first, empty, original

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        values = (torch.arange(256, dtype=torch.float32).reshape(16, 16) % 5 / 32,)
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
