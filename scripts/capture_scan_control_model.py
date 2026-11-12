#!/usr/bin/env python3
"""Capture learned state and dependent branch/loop scan calls with all results."""

import argparse
import math
import sys
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2m.examples.scan_control_kernels import loop_scan, segmented_scan


def capture(out, gcd=False):
    out = Path(out).resolve()
    segmented_name, loop_name = "t2m_demo::segmented_scan", "t2m_demo::loop_scan"

    @triton_op(segmented_name, mutates_args={})
    def segments(
        x: torch.Tensor, heads: torch.Tensor, n: int, reverse: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result, flags = torch.empty_like(x), torch.empty_like(heads)
        if n:
            wrap_triton(segmented_scan)[((n + 7) // 8,)](
                x, heads, result, flags, n, B=8, REVERSE=reverse
            )
        return result, flags

    @segments.register_kernel("cpu")
    def reference_segments(x, heads, n, reverse):
        result, flags = x.clone(), heads.clone()
        for start in range(0, n, 8):
            order = list(range(start, min(start + 8, n)))
            if reverse:
                order.reverse()
            previous, flag = 0, 0
            for index in order:
                value = (
                    int(x[index]) if int(heads[index]) else previous + int(x[index])
                ) & ((1 << 64) - 1)
                if value & (1 << 63):
                    value -= 1 << 64
                flag |= int(heads[index])
                result[index], flags[index] = value, flag
                previous = value
        return result, flags

    @triton_op(loop_name, mutates_args={})
    def loops(x: torch.Tensor, n: int, gcd: bool) -> torch.Tensor:
        result = torch.empty_like(x)
        if n:
            wrap_triton(loop_scan)[((n + 7) // 8,)](
                x, result, n, B=8, REVERSE=False, GCD=gcd
            )
        return result

    @loops.register_kernel("cpu")
    def reference_loops(x, n, gcd):
        result = x.clone()
        for start in range(0, n, 8):
            previous = 0 if gcd else 1
            for index in range(start, min(start + 8, n)):
                previous = (
                    math.gcd(previous, int(x[index]))
                    if gcd
                    else (previous * int(x[index])) & ((1 << 64) - 1)
                )
                result[index] = (
                    previous - (1 << 64) if previous & (1 << 63) else previous
                )
        return result

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(19, dtype=torch.int64) % 9 + 1, requires_grad=False
            )
            self.bias = torch.nn.Parameter(
                torch.arange(19, dtype=torch.int64) % 7 - 3, requires_grad=False
            )
            heads = torch.tensor(([0, 1, 0, 0, 1, 0, 0, 0] * 3)[:19], dtype=torch.int32)
            self.register_buffer("heads", heads)

        def forward(self, x):
            original = x + self.weight
            first, flags = segments(original, self.heads, 19, False)
            loop = loops(first, 19, gcd)
            second, second_flags = segments(loop, flags, 19, True)
            empty = loops(x[:0], 0, gcd)
            return loop + self.bias, first, flags, second, second_flags, empty, original

    old = {name: KERNEL_REGISTRY.get(name) for name in (segmented_name, loop_name)}
    try:
        for name in old:
            KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        values = torch.arange(19, dtype=torch.int64) % 11 + 2
        values[0] = (1 << 62) + 3
        inputs = (values,)
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
        for name, previous in old.items():
            if previous is None:
                KERNEL_REGISTRY.pop(name, None)
            else:
                KERNEL_REGISTRY[name] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gcd", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, args.gcd))
