#!/usr/bin/env python3
"""Capture actual root/view operands and complete functional state versions."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, dtype="i64"):
    out = Path(out).resolve()
    if out.exists(): raise FileExistsError(out)
    source = Path(__file__).resolve().parents[1] / "t2m/examples/model_view_kernels.py"
    loader = importlib.util.spec_from_file_location("source_view_kernels", source)
    kernels = importlib.util.module_from_spec(loader); loader.loader.exec_module(kernels)
    storage = {"i64": torch.int64, "f32": torch.float32, "bf16": torch.bfloat16}[dtype]
    qualname = "t2m_demo::source_view_update"

    @triton_op(qualname, mutates_args={})
    def update(owner: torch.Tensor, view: torch.Tensor, rows: int, cols: int,
               s0: int, s1: int, origin: int, capacity: int, steps: int,
               bias: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        result = owner.contiguous().clone()
        selected = result.as_strided((rows, cols), (s0, s1), origin)
        values = torch.empty((rows, cols), dtype=owner.dtype, device=owner.device)
        wrap_triton(kernels.update_source_view)[((rows + 7) // 8, (cols + 3) // 4)](
            result, selected, values, rows, cols, s0, s1, steps, bias, BM=8, BN=4)
        return result, values

    @update.register_kernel("cpu")
    def reference(owner, view, rows, cols, s0, s1, origin, capacity, steps, bias, count):
        if (not owner.is_contiguous() or owner.storage_offset() != 0 or owner.numel() != capacity
                or tuple(view.shape) != (rows, cols) or tuple(view.stride()) != (s0, s1)
                or view.storage_offset() != origin or count != rows * cols
                or owner.untyped_storage()._cdata != view.untyped_storage()._cdata):
            raise ValueError("source view differs from its actual owning storage relation")
        result = owner.clone()
        selected = result.as_strided((rows, cols), (s0, s1), origin)
        for _ in range(steps): selected.add_(bias)
        return result, selected.clone()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            weight = torch.arange(96, dtype=torch.int64).reshape(6, 16) % 13 - 5
            if dtype == "i64": weight += 2**53 + 1
            self.weight = torch.nn.Parameter(weight.to(storage), requires_grad=False)
            self.bias = torch.nn.Parameter((torch.arange(96).reshape(6, 16) % 5 - 2).to(storage), requires_grad=False)

        def forward(self, x):
            initial = x + self.weight
            first_view = initial.t()[1::2, :]
            first, observed = update(initial, first_view, 8, 6, 2, 16, 1, 96, 2, 5, 48)
            next_view = first[::2, 2::3]
            second, later = update(first, next_view, 3, 5, 32, 3, 2, 96, 1, -3, 15)
            empty_view = first.t()[:0, :]
            empty, none = update(first, empty_view, 0, 6, 1, 16, 0, 96, 3, 2, 0)
            return second + self.bias, first, observed, later, empty, none, first_view

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        values = ((torch.arange(96).reshape(6, 16) % 7).to(storage),)
        out.mkdir(parents=True, exist_ok=False)
        converted = m2m.convert(model, values, backend="fx_importer", capture_trace=True,
                                weights_path=str(out / "weights.safetensors"))
        if not converted.ok: raise ValueError(converted.diagnostics)
        write_bundle(model, values, out, capture_regions=False, capture_trace=True,
                     source_path=Path(__file__), conversion_result=converted)
    finally:
        if previous is None: KERNEL_REGISTRY.pop(qualname, None)
        else: KERNEL_REGISTRY[qualname] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("i64", "f32", "bf16"), default="i64")
    args = parser.parse_args()
    print(capture(args.out, args.dtype))
