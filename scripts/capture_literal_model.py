"""Capture a real typed source literal and an independently supplied runtime input."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.library import triton_op, wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


def capture(out, dtype="f32"):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    path = Path(__file__).resolve().parents[1] / "t2m/examples/literal_kernels.py"
    spec = importlib.util.spec_from_file_location("t2m_literal_source", path)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    if dtype in ("f16", "bf16", "f32", "f64"):
        bits = 16 if dtype in ("f16", "bf16") else int(dtype[1:])
        patterns = {"f16": [0, 0x8000, 1, 0x8001, 0x7c01, 0xfc35, 0x7e55, 0x3c00],
            "bf16": [0, 0x8000, 1, 0x8001, 0x7f81, 0xffa5, 0x7fc2, 0x3f80],
            "f32": [0, 0x80000000, 1, 0x80000001, 0x7f800001, 0xff800035, 0x7fc00012, 0x3f800000],
            "f64": [0, 1 << 63, 1, (1 << 63) | 1, 0x7ff0000000000001, 0xfff0000000000035, 0x7ff8000000000012, 0x3ff0000000000000]}
        raw = np.resize(np.asarray(patterns[dtype], dtype=getattr(np, f"uint{bits}")), 17).view(getattr(np, f"int{bits}"))
        literal = torch.from_numpy(raw.copy()).view(torch.bfloat16 if dtype == "bf16" else getattr(torch, f"float{bits}"))
    elif dtype == "i1":
        literal = torch.arange(17) % 2 == 0
    else:
        literal = (torch.arange(17) - 8).to(getattr(torch, "int" + dtype[1:]))
        if dtype == "i64":
            literal[0], literal[1] = 2**60 + 1, -(2**60 + 3)
    qualname = "t2m_demo::literal_copy"

    @triton_op(qualname, mutates_args={})
    def implementation(literal: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(literal)
        wrap_triton(kernels.literal_copy)[(3,)](literal, out, N=17, B=8)
        return out

    @implementation.register_kernel("cpu")
    def reference(literal):
        return literal.clone()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.literal = literal
        def forward(self, x):
            return implementation(self.literal), x.clone()

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname, description="Typed source literal qualification")
    try:
        model, values = Model().eval(), (torch.tensor([7., -3., 2.]),)
        out.mkdir(parents=True)
        result = m2m.convert(model, values, backend="fx_importer", weights_path=str(out / "weights.safetensors"), capture_trace=True)
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, values, out, conversion_result=result, capture_trace=True, capture_regions=False, source_path=Path(__file__))
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(qualname, None)
        else:
            KERNEL_REGISTRY[qualname] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("f16", "bf16", "f32", "f64", "i1", "i8", "i16", "i32", "i64"), default="f32")
    args = parser.parse_args()
    print(capture(args.out, args.dtype))
