"""Capture real BF16 model calls with exact words, learned state and all results."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.library import triton_op, wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


def capture(out, mode="observe", *, finite=False, lifted=False):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    path = Path(__file__).resolve().parents[1] / "t2m/examples" / ("bfloat_kernels.py" if mode == "observe" else "tensor_kernels.py")
    spec = importlib.util.spec_from_file_location("t2m_bfloat_sources", path)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::bfloat_" + mode
    if mode == "observe":
        @triton_op(qualname, mutates_args={})
        def implementation(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            out = torch.empty_like(x)
            words = torch.empty_like(x, dtype=torch.int16)
            value = torch.empty_like(x, dtype=torch.float32)
            wrap_triton(kernels.bfloat_observe)[(3,)](x, weight, out, words, value, N=17, B=8)
            return out, words, value

        @implementation.register_kernel("cpu")
        def reference(x, weight):
            selected = torch.where(torch.arange(x.numel()) % 2 == 0, x.view(torch.int16), weight.view(torch.int16))
            return selected.view(torch.bfloat16).clone(), selected.clone(), x.float() + weight.float()

        def raw(values):
            return torch.from_numpy(np.asarray(values, np.uint16).view(np.int16).copy()).view(torch.bfloat16)

        patterns = np.array([0, 0x8000, 1, 0x807f, 0x7f81, 0xffa5, 0x7f80, 0xff80, 0x3f80, 0xbf80,
            0x7fc2, 0xff81, 2, 0x8001, 0x4000, 0xc000, 0x7f7f], np.uint16)
        if finite:
            patterns[(patterns & 0x7f80) == 0x7f80] = 0x4080

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if lifted:
                    self.weight, self.cache = raw(np.roll(patterns, 3)), raw(patterns[::-1])
                else:
                    self.weight = torch.nn.Parameter(raw(np.roll(patterns, 3)), requires_grad=False)
                    self.register_buffer("cache", raw(patterns[::-1]))

            def forward(self, x):
                first = implementation(x, self.weight)
                second = implementation(first[0], self.cache)
                return (*first, *second)

        values = (raw(patterns),)
    elif mode == "matmul":
        @triton_op(qualname, mutates_args={})
        def implementation(x: torch.Tensor, weight: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(initial)
            wrap_triton(kernels.accumulated_matmul)[(1,)](x, weight, initial, out, M=16, N=16, K=16)
            return out

        @implementation.register_kernel("cpu")
        def reference(x, weight, initial):
            return initial + x.float() @ weight.float()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                weight = (torch.arange(256).reshape(16, 16) % 5 - 2).to(torch.bfloat16)
                initial = (torch.arange(256).reshape(16, 16) % 11 - 5).to(torch.float32)
                self.weight = weight if lifted else torch.nn.Parameter(weight, requires_grad=False)
                self.initial = initial if lifted else torch.nn.Parameter(initial, requires_grad=False)

            def forward(self, x):
                return implementation(x, self.weight, self.initial)

        values = ((torch.arange(256).reshape(16, 16) % 7 - 3).to(torch.bfloat16),)
    else:
        raise ValueError("unknown BF16 model mode")
    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname, description="BF16 model interface qualification")
    try:
        model = Model().eval()
        out.mkdir(parents=True)
        result = m2m.convert(model, values, backend="fx_importer", weights_path=str(out / "weights.safetensors"), capture_trace=True)
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, values, out, capture_regions=False, capture_trace=True, source_path=Path(__file__), conversion_result=result)
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(qualname, None)
        else:
            KERNEL_REGISTRY[qualname] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("observe", "matmul"), default="observe")
    parser.add_argument("--finite", action="store_true", help="finite observation data for legacy archive qualification")
    parser.add_argument("--lifted", action="store_true", help="capture source literals instead of registered state")
    args = parser.parse_args()
    print(capture(args.out, args.mode, finite=args.finite, lifted=args.lifted))
