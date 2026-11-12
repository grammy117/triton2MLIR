#!/usr/bin/env python3
"""Capture complete model calls with decoded word arithmetic and descriptors."""

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
from t2m.examples.assembly_descriptor_reference import update


def capture(out, *, dtype="i64"):
    out = Path(out).resolve()

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded

    kernels = module(
        "descriptor_assembly_model_kernel",
        ROOT / "t2m/examples/assembly_descriptor_kernels.py",
    )
    word_program = module(
        "descriptor_assembly_model_words",
        ROOT / "adapters/ptx/examples/model_reference.py",
    )
    qualname = "t2m_demo::assembly_descriptor_state"
    width = 64 if dtype == "i64" else 32

    @triton_op(qualname, mutates_args={})
    def assembly_descriptor_state(
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
        shift: int,
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
            wrap_triton(kernels.assembly_descriptor_words)[
                ((m + 1) // 2, (n + 15) // 16)
            ](
                first,
                second,
                m,
                n,
                steps,
                capacity,
                shift,
                ASM=word_program.program(width),
                WORDS64=width == 64,
            )
        return result

    @assembly_descriptor_state.register_kernel("cpu")
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
        shift,
    ):
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
                shift=shift,
                words=True,
            )
        )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            pattern = [
                0,
                1,
                -1,
                (1 << (width - 1)) - 1,
                -(1 << (width - 1)),
                (1 << 53) + 1 if width == 64 else (1 << 24) + 1,
            ]
            root = np.resize(
                np.asarray(pattern, dtype="int64" if width == 64 else "int32"), 256
            ).reshape(16, 16)
            self.weight = torch.nn.Parameter(
                torch.from_numpy(root), requires_grad=False
            )
            storage = torch.int64 if width == 64 else torch.int32
            self.bias = torch.nn.Parameter(
                torch.arange(256, dtype=storage).reshape(16, 16) % 7 - 3,
                requires_grad=False,
            )

        def forward(self, x):
            original = x ^ self.weight
            first = assembly_descriptor_state(
                original, 2, 13, 2, 256, 8, 16, 24, 1, 24, 1, 3
            )
            second = assembly_descriptor_state(
                first, 5, 19, 3, 256, 8, 8, 24, 1, 24, 1, 33
            )
            empty = assembly_descriptor_state(
                x[:0], 0, 13, 2, 0, 8191, 8191, 24, 1, 24, 1, -1
            )
            return second + self.bias, first, empty, original

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (
            torch.arange(
                256, dtype=torch.int64 if width == 64 else torch.int32
            ).reshape(16, 16)
            % 5,
        )
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(
            model,
            inputs,
            backend="fx_importer",
            weights_path=str(out / "weights.safetensors"),
            capture_trace=True,
        )
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(
            model,
            inputs,
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
    parser.add_argument("--dtype", choices=("i32", "i64"), default="i64")
    args = parser.parse_args()
    print(capture(args.out, dtype=args.dtype))
