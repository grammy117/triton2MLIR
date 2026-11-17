#!/usr/bin/env python3
"""Real broadcasts/slices feeding dependent Triton kernels with derived extents."""

import argparse
import sys
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2m.examples.live_scalar_kernels import live_extent_copy


def capture(out):
    name = "t2m_demo::derived_copy"

    @triton_op(name, mutates_args={})
    def copy(x: torch.Tensor, n: int) -> torch.Tensor:
        out = torch.empty_like(x)
        wrap_triton(live_extent_copy)[((n + 7) // 8,)](x, n, out, B=8)
        return out

    @copy.register_kernel("cpu")
    def reference(x, n):
        assert x.numel() == n
        return x.clone()

    @copy.register_fake
    def fake(x, n):
        return torch.empty_like(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor(17, dtype=torch.int32), requires_grad=False
            )

        def forward(self, x, y):
            original = x + y + self.weight
            trimmed = original[1::2]
            first = copy(trimmed, trimmed.shape[0])
            second = copy(first, first.shape[0])
            return second, first, original, trimmed, x, y, self.weight

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (
            torch.arange(13, dtype=torch.int32),
            torch.arange(13, dtype=torch.int32) * -3,
        )
        rows = torch.export.Dim("rows", min=4, max=31)
        program = torch.export.export(
            model, inputs, dynamic_shapes={"x": {0: rows}, "y": {0: rows}}
        )
        out = Path(out).resolve()
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(
            program,
            inputs,
            backend="fx_importer",
            capture_trace=True,
            weights_path=str(out / "weights.safetensors"),
        )
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(
            model,
            inputs,
            out,
            capture_trace=True,
            capture_regions=False,
            conversion_result=result,
            source_path=Path(__file__),
        )
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(name, None)
        else:
            KERNEL_REGISTRY[name] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(capture(parser.parse_args().out))
