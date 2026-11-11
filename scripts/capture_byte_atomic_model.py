#!/usr/bin/env python3
"""Capture selected byte atomics with learned state and all tuple references."""

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


def capture(out, *, width=32, cas=False, typed_root=False):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    root = Path(__file__).resolve().parents[1]
    kernels = load("byte_atomic_model_kernels", root / "t2m/examples/byte_atomic_kernels.py")
    oracle = load("byte_atomic_model_reference", root / "t2m/examples/byte_atomic_state_reference.py")
    qualname = "t2m_demo::byte_atomic_state"
    dtype = torch.int64 if typed_root else torch.int8

    @triton_op(qualname, mutates_args={})
    def byte_atomic_state(a: torch.Tensor, b: torch.Tensor, n: int, steps: int,
                          capacity: int, choice: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first, second = a.contiguous().clone(), b.contiguous().clone()
        history = torch.empty((steps, n), dtype=getattr(torch, f"int{width}"), device=a.device)
        wrap_triton(kernels.byte_atomic_state)[(1,)](first.view(-1), second.view(-1), history,
            n, steps, capacity, choice, B=4, WIDTH=width, CAS=cas, OP="xor", WHILE=True, NARROW=True, CARRY=True)
        return first, second, history

    @byte_atomic_state.register_kernel("cpu")
    def reference(a, b, n, steps, capacity, choice):
        if a.numel() != capacity or b.numel() != capacity:
            raise ValueError("owning model state differs from the source capacity")
        results = oracle.reference(a.detach().contiguous().numpy(), b.detach().contiguous().numpy(),
            n, steps, choice, width=width, cas=cas, byte_order="little", carry=True)
        return tuple(torch.from_numpy(value) for value in results)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            byte_count = 64 * (8 if typed_root else 1)
            for name, shift in (("a", 211), ("b", 97)):
                raw = ((torch.arange(byte_count, dtype=torch.int64) * 73 + shift) & 255).to(torch.uint8)
                self.register_parameter(name, torch.nn.Parameter(raw.view(dtype).reshape(8, 8), requires_grad=False))

        def forward(self, x):
            a0, b0, first = byte_atomic_state(x + self.a, self.b, 3, 3, 64, True)
            a1, b1, second = byte_atomic_state(a0, b0, 4, 2, 64, False)
            empty_a, empty_b, empty_history = byte_atomic_state(x[:0], x[:0], 0, 0, 0, False)
            return a1, b1, first, second, empty_a, empty_b, empty_history

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model, values = Model().eval(), (torch.zeros((8, 8), dtype=dtype),)
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
    parser.add_argument("--width", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--cas", action="store_true")
    parser.add_argument("--typed-root", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, width=args.width, cas=args.cas, typed_root=args.typed_root))
