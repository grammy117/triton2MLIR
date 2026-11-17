#!/usr/bin/env python3
"""Capture functional model calls with carried addresses and independent CPU math."""

import argparse
import sys
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2m.examples.integer_address_region_kernels import integer_address_state


def capture(out, width=64, mode="alternating"):
    out = Path(out).resolve()
    if width not in (32, 64) or mode not in ("for", "while", "branch", "alternating"):
        raise ValueError("the model requires i32/i64 words and a declared control form")
    storage = torch.int32 if width == 32 else torch.int64
    name = f"t2m_demo::integer_address_{mode}_{width}"

    @triton_op(name, mutates_args={})
    def address_state(x: torch.Tensor, y: torch.Tensor, initial: torch.Tensor,
                      n: int, capacity: int, steps: int, origin: int, stride: int,
                      choice: int) -> tuple[torch.Tensor, torch.Tensor]:
        state = initial.clone()
        distance = torch.empty((n,), dtype=torch.int64, device=x.device)
        if n:
            wrap_triton(integer_address_state)[((n + 7) // 8,)](
                x, y, state, distance, n, capacity, steps, origin, stride, choice,
                B=8, WIDTH=width, MODE=mode)
        return state, distance

    @address_state.register_kernel("cpu")
    def reference(x, y, initial, n, capacity, steps, origin, stride, choice):
        assert x.numel() == y.numel() == capacity and initial.numel() == n
        result = initial.clone()
        root = y if mode == "branch" and choice <= 0 else x
        position = origin
        for step in range(steps):
            if n:
                assert 0 <= position and position + n <= capacity
                result = torch.bitwise_xor(result, root[position:position + n])
            position += stride
            if mode == "alternating":
                root = x if (step + choice) % 2 == 0 else y
        distance = torch.full((n,), steps * stride * (width // 8), dtype=torch.int64)
        return result, distance

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(128, dtype=storage) * -3571 + 17, requires_grad=False)
            self.initial = torch.nn.Parameter(torch.arange(19, dtype=storage) * 37 - 91, requires_grad=False)
            self.bias = torch.nn.Parameter(torch.arange(19, dtype=storage) % 5 - 2, requires_grad=False)

        def forward(self, x):
            original = x + self.weight
            first, first_distance = address_state(original, self.weight, self.initial, 19, 128, 4, 80, -3, 1)
            follow, follow_distance = address_state(original, self.weight, first, 19, 128, 2, 31, -8, -1)
            empty, empty_distance = address_state(x[:0], self.weight[:0], self.initial[:0], 0, 0, 3, 256, 8, -2)
            zero, zero_distance = address_state(x[:0], self.weight[:0], first, 19, 0, 0, -256, -8, 0)
            return (follow, follow_distance, first, first_distance, empty, empty_distance,
                    zero, zero_distance, original, first + self.bias, self.initial)

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (torch.arange(128, dtype=storage) * 7919 + (1 << (29 if width == 32 else 58)),)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
                             weights_path=str(out / "weights.safetensors"))
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, inputs, out, capture_regions=False, capture_trace=True,
                     source_path=Path(__file__), conversion_result=result)
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(name, None)
        else:
            KERNEL_REGISTRY[name] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", choices=(32, 64), type=int, default=64)
    parser.add_argument("--mode", choices=("for", "while", "branch", "alternating"), default="alternating")
    args = parser.parse_args()
    print(capture(args.out, args.width, args.mode))
