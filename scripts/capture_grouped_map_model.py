#!/usr/bin/env python3
"""Capture an explicit initial source-layout contract, not final GPU equivalence."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, *, dtype="i64", warps=4, control=False):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if control and dtype != "i64":
        raise ValueError("the control example uses integer bit operations")
    path = Path(__file__).resolve().parents[1] / "t2m/examples/grouped_map_kernels.py"
    spec = importlib.util.spec_from_file_location("source_grouped_capture_kernels", path)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::source_grouped_map"
    scalar_type = {"i64": torch.int64, "f32": torch.float32}[dtype]
    n, tile = 300, 256

    @triton_op(qualname, mutates_args={})
    def implementation(x: torch.Tensor, y: torch.Tensor, n: int, source_target: str,
                       source_warps: int, source_threads: int, source_ctas: int,
                       control: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if source_target != "cuda:80" or source_threads != 32 or source_ctas != 1:
            raise ValueError("unqualified example source specialization")
        first, second = torch.empty_like(x), torch.empty_like(x)
        if control:
            wrap_triton(kernels.grouped_control_map)[((n + tile - 1) // tile,)](
                x, y, first, second, n, B=tile, num_warps=source_warps)
        else:
            wrap_triton(kernels.grouped_map)[((n + tile - 1) // tile,)](
                x, y, first, second, n, B=tile, PACK=2, MATRIX=False,
                num_warps=source_warps)
        return first, second

    @implementation.register_kernel("cpu")
    def reference(x, y, n, source_target, source_warps, source_threads, source_ctas, control):
        # This independent example oracle defines the DECLARED initial-stage
        # specialization. It does not read frontend witnesses or emitted IR.
        if (source_target != "cuda:80" or source_warps not in (2, 4)
                or source_threads != 32 or source_ctas != 1 or x.numel() != n):
            raise ValueError("unqualified example source specialization")
        padding = (-n) % tile
        a = torch.nn.functional.pad(x, (0, padding)).reshape(-1, 2, source_warps * source_threads)
        b = torch.nn.functional.pad(y, (0, padding)).reshape_as(a)
        a0, a1, b0, b1 = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        if control:
            divisor = torch.where(b0 != 0, b0, torch.ones_like(b0))
            value = torch.where(b0 != 0, torch.div(a0, divisor, rounding_mode="trunc"), a1)
            for step in range(3):
                value = torch.where((b1 & 3) > step, value + a1, value)
            first = torch.stack((value, value + b1), dim=1)
        else:
            first = torch.stack((a0 + a1, a1 - a0), dim=1)
        second = torch.stack((a0 + b0, a1 - b1), dim=1)
        return first.flatten()[:n], second.flatten()[:n]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            values = torch.tensor([0, -2, 4, 1, -1, 0, 2], dtype=scalar_type).repeat(43)[:n]
            self.weight = torch.nn.Parameter(values, requires_grad=False)

        def forward(self, x):
            parameters = (n, "cuda:80", warps, 32, 1, control)
            first = implementation(x, self.weight, *parameters)
            second = implementation(first[0], self.weight, *parameters)
            return *first, *second

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        x = torch.arange(n, dtype=scalar_type) * 3 - 17
        if dtype == "i64":
            x += 1 << 56
        model, values = Model().eval(), (x,)
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
    parser.add_argument("--dtype", choices=("i64", "f32"), default="i64")
    parser.add_argument("--warps", type=int, choices=(2, 4), default=4)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, dtype=args.dtype, warps=args.warps, control=args.control))
