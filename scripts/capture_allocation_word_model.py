#!/usr/bin/env python3
"""Capture real model calls whose differently typed kernel arguments alias."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = (
        Path(__file__).resolve().parents[1] / "t2m/examples/allocation_word_kernels.py"
    )
    spec = importlib.util.spec_from_file_location(
        "allocation_word_model_kernels", source
    )
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::allocation_words"

    @triton_op(qualname, mutates_args={})
    def allocation_words(
        state: torch.Tensor,
        n: int,
        steps: int,
        capacity: int,
        read_origin: int,
        write_origin: int,
    ) -> torch.Tensor:
        updated = state.contiguous().clone()
        flat = updated.view(-1)
        wrap_triton(kernels.allocation_word_updates)[((n + 7) // 8,)](
            flat[read_origin:].view(torch.float32),
            flat[write_origin:],
            n,
            steps,
            capacity,
            B=8,
            WIDTH=32,
            WHILE=True,
        )
        return updated

    @allocation_words.register_kernel("cpu")
    def reference(state, n, steps, capacity, read_origin, write_origin):
        if state.numel() != capacity:
            raise ValueError("word model allocation capacity differs from source")
        updated = state.contiguous().clone()
        flat = updated.view(-1)
        for start in range(0, n, 8):
            end = min(n, start + 8)
            for _ in range(steps):
                value = flat[read_origin + start : read_origin + end].clone()
                flat[write_origin + start : write_origin + end] = value ^ 1
                forwarded = flat[read_origin + start : read_origin + end].clone()
                flat[write_origin + start : write_origin + end] = forwarded ^ 2
        return updated

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            words = torch.tensor(
                [
                    0,
                    1,
                    -2147483648,
                    0x7F800001,
                    0x7FC00035,
                    -8388608,
                    -4194250,
                    0x7F7FFFFF,
                ],
                dtype=torch.int32,
            )
            self.weight = torch.nn.Parameter(
                words.repeat(12).reshape(8, 12), requires_grad=False
            )
            self.bias = torch.nn.Parameter(
                torch.arange(96, dtype=torch.int32).reshape(8, 12) % 7 - 3,
                requires_grad=False,
            )

        def forward(self, x):
            first = allocation_words(x + self.weight, 7, 2, 96, 8, 10)
            second = allocation_words(first, 31, 3, 96, 8, 8)
            empty = allocation_words(x[:0], 0, 2, 0, 511, -512)
            return second + self.bias, first, empty

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model, values = (
            Model().eval(),
            (torch.arange(96, dtype=torch.int32).reshape(8, 12) % 5,),
        )
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
