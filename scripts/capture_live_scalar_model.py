#!/usr/bin/env python3
"""Capture live singleton controls or actual exported shape SSA at kernel calls."""

import argparse
import sys
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2m.examples.live_scalar_kernels import live_extent_copy, live_scalar_values


def capture(out, dtype="i64", *, extent=False):
    out = Path(out).resolve()
    types = {"i1": torch.bool, "i8": torch.int8, "i16": torch.int16, "i32": torch.int32,
             "i64": torch.int64, "f32": torch.float32, "f64": torch.float64}
    if dtype not in types or extent and dtype != "i32":
        raise ValueError("the shape model uses i32; controls use a declared runtime scalar type")
    name = "t2m_demo::live_extent" if extent else "t2m_demo::live_scalar_" + dtype
    kind = "bool" if dtype == "i1" else "float" if dtype.startswith("f") else "integer"
    storage = torch.int32 if dtype == "i1" else types[dtype]
    if extent:
        @triton_op(name, mutates_args={})
        def operation(x: torch.Tensor, n: int) -> torch.Tensor:
            out = torch.empty_like(x)
            wrap_triton(live_extent_copy)[((n + 7) // 8,)](x, n, out, B=8)
            return out

        @operation.register_kernel("cpu")
        def reference(x, n):
            assert x.numel() == n
            return x.clone()
    else:
        @triton_op(name, mutates_args={})
        def operation(x: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            wrap_triton(live_scalar_values)[((x.numel() + 7) // 8,)](
                x, control.item(), out, N=x.numel(), B=8, KIND=kind)
            return out

        @operation.register_kernel("cpu")
        def reference(x, control):
            value = control.item()
            return (x if value else -x).clone() if kind == "bool" else x + value if kind == "float" else x ^ value

    # Fake execution specifies only the real result type/shape. CPU reference
    # execution above supplies goldens; no fake value enters compiler facts.
    @operation.register_fake
    def fake(x, argument):
        return torch.empty_like(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(False if dtype == "i1" else
                .25 if kind == "float" else 17, dtype=types[dtype]), requires_grad=False)

        def forward(self, x, control=None):
            if extent:
                original = x + self.weight
                first = operation(original, x.shape[0])
                second = operation(first, first.shape[0])
                empty = operation(x[:0], 0)
                return second, first, empty, original, self.weight
            first = operation(x, self.weight)
            second = operation(first, control)
            return second, first, x, control, self.weight

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        x = torch.arange(13, dtype=storage)
        x = x * .125 - .75 if kind == "float" else x * 7 - 43
        if dtype == "i64":
            x += 1 << 58
        inputs = (x,) if extent else (x, torch.tensor([[True if dtype == "i1" else
                    .75 if kind == "float" else -91]], dtype=types[dtype]))
        original = torch.export.export(model, inputs,
            dynamic_shapes={"x": {0: torch.export.Dim("n", min=2, max=31)}}) if extent else model
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(original, inputs, backend="fx_importer", capture_trace=True,
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
    parser.add_argument("--dtype", choices=("i1", "i8", "i16", "i32", "i64", "f32", "f64"), default="i64")
    parser.add_argument("--extent", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, args.dtype, extent=args.extent))
