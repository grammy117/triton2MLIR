#!/usr/bin/env python3
"""Capture a real model with typed words and mutable private byte-cache values."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, *, typed_root=False):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = Path(__file__).resolve().parents[1] / "t2m/examples/byte_buffer_kernels.py"
    spec = importlib.util.spec_from_file_location("byte_model_kernels", source)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::byte_cache"

    @triton_op(qualname, mutates_args={})
    def byte_cache(
        state: torch.Tensor,
        n: int,
        steps: int,
        capacity: int,
        origin: int,
        stride: int,
        advance: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        updated = state.contiguous().clone()
        words = torch.empty((steps, n), dtype=torch.int32, device=state.device)
        wrap_triton(kernels.byte_buffer_updates)[((n + 3) // 4,)](
            updated.view(-1),
            words,
            n,
            steps,
            capacity,
            origin,
            stride,
            advance,
            B=4,
            TYPE="f32",
            WIDTH=32,
            SCALAR=False,
            NARROW=True,
        )
        return updated, words

    @byte_cache.register_kernel("cpu")
    def reference(state, n, steps, capacity, origin, stride, advance):
        if state.numel() != capacity:
            raise ValueError("source byte capacity differs from its model value")
        memory = [int(v) & 255 for v in state.contiguous().view(torch.uint8).view(-1).tolist()]
        root_bytes = state.element_size()
        words = torch.empty((steps, n), dtype=torch.int32)
        for first in range(0, n, 4):
            for iteration in range(steps):
                snapshots = []
                for lane in range(first, min(n, first + 4)):
                    address = (origin + lane * stride) * root_bytes + iteration * advance * 4
                    if address < 0 or address + 4 > len(memory):
                        raise ValueError("source byte view escapes model storage")
                    value = (
                        int.from_bytes(bytes(memory[address : address + 4]), "little")
                        ^ 3
                    )
                    snapshots.append((lane, address, value))
                for _, address, value in snapshots:
                    memory[address : address + 4] = value.to_bytes(4, "little")
                for _, address, _ in snapshots:
                    memory[address + 1] = 0x5A
                for lane, address, _ in snapshots:
                    value = int.from_bytes(
                        bytes(memory[address : address + 4]), "little"
                    )
                    words[iteration, lane] = value if value < 2**31 else value - 2**32
        updated = torch.tensor(
            [v if v < 128 else v - 256 for v in memory], dtype=torch.int8
        )
        return updated.view(state.dtype).reshape(state.shape), words

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            cache = (
                ((torch.arange(256, dtype=torch.int64) * 73 + 211) & 255)
                .to(torch.int32 if typed_root else torch.int8)
                .reshape(16, 16)
            )
            self.cache = torch.nn.Parameter(cache, requires_grad=False)

        def forward(self, x):
            first, words0 = byte_cache(x + self.cache, 7, 2, 256, 9, 24, 1)
            second, words1 = byte_cache(first, 13, 1, 256, 7, 16, 0)
            empty, empty_words = byte_cache(x[:0], 0, 3, 0, 8191, -128, -64)
            return second, words0, words1, empty, empty_words

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model, values = Model().eval(), (torch.zeros((16, 16), dtype=torch.int32 if typed_root else torch.int8),)
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
    parser.add_argument("--typed-root", action="store_true", help="retain public i32 storage around byte views")
    args = parser.parse_args()
    print(capture(args.out, typed_root=args.typed_root))
