#!/usr/bin/env python3
"""Capture real scan triton_op models with independent eager PyTorch goldens."""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.library import triton_op, wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


def capture(out, *, kind="sum", reverse=False):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = Path(__file__).resolve().parents[1] / "t2m/examples/tensor_kernels.py"
    spec = importlib.util.spec_from_file_location("t2m_scan_kernels", source)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    if kind == "tuple":
        kernel = kernels.tuple_scan
        qualname = "t2m_demo::affine_scan"

        @triton_op(qualname, mutates_args={})
        def scan(a: torch.Tensor, b: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
            out_a, out_b = torch.empty_like(a), torch.empty_like(b)
            wrap_triton(kernel)[(1,)](a, b, out_a, out_b, R=2, S=8, AXIS=1, REVERSE=reverse)
            return out_a, out_b

        @scan.register_kernel("cpu")
        def reference(a, b, reverse=False):
            order = list(range(a.shape[1]))
            prefix, results = None, {}
            for index in reversed(order) if reverse else order:
                av, bv = a[:, index], b[:, index]
                prefix = (av, bv) if prefix is None else (av * prefix[0], av * prefix[1] + bv)
                results[index] = prefix
            return tuple(torch.stack([results[i][j] for i in order], dim=1) for j in range(2))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randint(-2, 3, (2, 8)).float())

            def forward(self, x):
                a, b = scan(x, self.weight, reverse=reverse)
                return a + b

        inputs = lambda: (torch.randint(0, 3, (2, 8)).float(),)
    else:
        kernel = kernels.scan_tiles
        qualname = "t2m_demo::scan"

        @triton_op(qualname, mutates_args={})
        def scan(x: torch.Tensor, reverse: bool = False, kind: int = 0) -> torch.Tensor:
            out = torch.empty_like(x)
            wrap_triton(kernel)[((x.numel() + 15) // 16,)](
                x, out, N=x.numel(), B=16, KIND=kind, REVERSE=reverse)
            return out

        @scan.register_kernel("cpu")
        def reference(x, reverse=False, kind=0):
            result = []
            for start in range(0, x.numel(), 16):
                tile = x[start:start + 16]
                if reverse:
                    tile = tile.flip((0,))
                tile = torch.cumsum(tile, 0) if kind == 0 else torch.cumprod(tile, 0)
                result.append(tile.flip((0,)) if reverse else tile)
            return torch.cat(result)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(53) * 0.2 + 0.9)

            def forward(self, x):
                return scan(x * self.weight, reverse=reverse, kind=0 if kind == "sum" else 1) + 0.25

        inputs = lambda: (torch.rand(53) * 0.4 + 0.8,)
    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(31)
        model, values = Model().eval(), inputs()
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(model, values, backend="fx_importer",
                             weights_path=str(out / "weights.safetensors"), capture_trace=True)
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
    parser.add_argument("--kind", choices=("sum", "product", "tuple"), default="sum")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, kind=args.kind, reverse=args.reverse))
