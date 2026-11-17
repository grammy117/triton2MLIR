#!/usr/bin/env python3
"""Capture actual scalar-tensor loop bounds and dependent functional state."""

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


def capture(out):
    out = Path(out).resolve()
    name = "t2m_demo::live_step_state"

    @triton_op(name, mutates_args={})
    def state(x: torch.Tensor, y: torch.Tensor, initial: torch.Tensor, steps: torch.Tensor,
              n: int, capacity: int, origin: int, stride: int, choice: int) -> tuple[torch.Tensor, torch.Tensor]:
        count = steps.item()
        if not 0 <= count <= 4:
            raise ValueError("step count outside the declared source contract")
        updated = initial.clone()
        distance = torch.empty((n,), dtype=torch.int64, device=x.device)
        wrap_triton(integer_address_state)[((n + 7) // 8,)](
            x, y, updated, distance, n, capacity, count, origin, stride, choice,
            B=8, WIDTH=64, MODE="alternating")
        return updated, distance

    @state.register_fake
    def fake(x, y, initial, steps, n, capacity, origin, stride, choice):
        return initial.clone(), torch.empty((n,), dtype=torch.int64, device=x.device)

    @state.register_kernel("cpu")
    def reference(x, y, initial, steps, n, capacity, origin, stride, choice):
        count = steps.item()
        if not 0 <= count <= 4:
            raise ValueError("step count outside the declared source contract")
        assert x.numel() == y.numel() == capacity and initial.numel() == n
        updated = initial.clone()
        pointer, root = origin, x
        for iteration in range(count):
            assert not n or 0 <= pointer and pointer + n <= capacity
            updated = torch.bitwise_xor(updated, root[pointer:pointer + n])
            pointer += stride
            root = x if (iteration + choice) % 2 == 0 else y
        return updated, torch.full((n,), count * stride * 8, dtype=torch.int64)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(128, dtype=torch.int64) * -3571 + 17, requires_grad=False)
            self.initial = torch.nn.Parameter(torch.arange(19, dtype=torch.int64) * 37 - 91, requires_grad=False)

        def forward(self, x, steps):
            original = x + self.weight
            first, distance = state(original, self.weight, self.initial, steps, 19, 128, 80, -3, 1)
            follow, follow_distance = state(original, self.weight, first, steps, 19, 128, 31, -2, -1)
            return follow, follow_distance, first, distance, original, self.initial, steps

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (torch.arange(128, dtype=torch.int64) * 7919 + (1 << 58), torch.tensor(3, dtype=torch.int32))
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
    args = parser.parse_args()
    print(capture(args.out))
