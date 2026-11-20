#!/usr/bin/env python3
"""Capture actual mutating Triton kernels with shared source view storage."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out, kind="overlap"):
    source = Path(__file__).resolve().parents[1] / "t2m/examples/mutating_views.py"
    loader = importlib.util.spec_from_file_location("mutating_view_kernel", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    matrix_views = kind == "permuted-base"
    update_views = kernels.update_matrix_views if matrix_views else kernels.update_views
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::mutating_views"

    @triton_op(name, mutates_args={"left", "right"})
    def update(left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        old_left, old_right = torch.empty_like(left), torch.empty_like(right)
        if matrix_views:
            wrap_triton(update_views)[(1,)](left, right, old_left, old_right, bias,
                left.shape[0], right.shape[0], left.stride(0), right.stride(0),
                left.shape[1], right.shape[1], left.stride(1), right.stride(1))
        else:
            wrap_triton(update_views)[(1,)](left, right, old_left, old_right, bias,
                left.numel(), right.numel(), left.stride(0), right.stride(0))
        return old_left, old_right

    @update.register_kernel("cpu")
    def reference(left, right, bias):
        before = left.clone(), right.clone()
        left.add_(bias)
        right.mul_(3)
        return before

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(2, dtype=torch.int64), requires_grad=False)

        def forward(self, x):
            state = x.permute(1, 2, 0).clone() if matrix_views else x.T.clone() if kind == "transposed-base" else x
            if kind == "overlap":
                left, right = state[1:-1], state[2:]
            elif kind == "stride":
                left, right = state[::2], state[1::2]
            elif kind in ("matrix", "transposed-base"):
                left, right = state[:, 1], state[:, -1]
            elif matrix_views:
                left, right = state[:, 1, :], state[:, -1, :]
            else:
                left, right = state[1:-1], state[1:-1]
            before = state.clone()
            a, b = update(left, right, self.bias)
            middle = state.clone()
            c, d = update(left, right, self.bias)
            return before, a, b, middle, c, d, state, self.bias

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        model = Model().eval()
        matrix = kind in ("matrix", "transposed-base")
        x = torch.arange(24).reshape(4, 6) if matrix else torch.arange(12)
        dynamic = ({0: torch.export.Dim("rows", min=2, max=7), 1: torch.export.Dim("cols", min=2, max=9)},) if matrix else (
            {0: torch.export.Dim("length", min=4, max=31)},)
        if matrix_views:
            x = torch.arange(60).reshape(3, 4, 5)
            dynamic = ({axis: torch.export.Dim(f"axis{axis}", min=2, max=maximum)
                        for axis, maximum in enumerate((4, 6, 7))},)
        exported = torch.export.export(model, (x,), dynamic_shapes=dynamic)
        converted = m2m.convert(exported, (x,), backend="fx_importer", capture_trace=True,
            weights_path=str(out / "weights.safetensors"))
        if not converted.ok:
            raise ValueError(converted.diagnostics)
        if converted.capture_trace["status"] != "complete":
            raise ValueError(converted.capture_trace["blockers"])
        write_bundle(model, (x,), out, conversion_result=converted, capture_trace=True,
                     capture_regions=False, source_path=Path(__file__))
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(name, None)
        else:
            KERNEL_REGISTRY[name] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=("overlap", "stride", "same", "matrix", "transposed-base", "permuted-base"), default="overlap")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(capture(args.out, args.kind))
