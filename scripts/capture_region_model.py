#!/usr/bin/env python3
"""Capture dependent partial-tile sum and coupled tuple-reduction recurrences."""

import argparse
import importlib.util
import sys
from pathlib import Path

import m2m
import torch
import triton
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton


def capture(out):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    source = Path(__file__).resolve().parents[1] / "t2m/examples/region_tile_kernels.py"
    loader = importlib.util.spec_from_file_location("region_source", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    name = "t2m_demo::region_partition"

    @triton_op(name, mutates_args={})
    def transform(x: torch.Tensor, bias: torch.Tensor, n: int, steps: int, kind: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        result, seen = torch.empty_like(x), torch.empty_like(x)
        wrap_triton(kernels.region_partition)[(triton.cdiv(n, 8),)](x, bias, result, seen, n, steps, B=8,
            MODE="zero-reduce" if kind == 0 else "tuple")
        return result, seen

    @transform.register_kernel("cpu")
    def reference(x, bias, n, steps, kind):
        assert x.numel() == n
        def signed(word):
            word &= (1 << 64) - 1
            return word - (1 << 64) if word & (1 << 63) else word

        results, snapshots = [], []
        for start in range(0, n, 8):
            length = min(8, n - start)
            valid = torch.arange(8) < length
            word = torch.zeros(8, dtype=torch.int64)
            word[:length] = x[start:start + length]
            word = word ^ bias[0]
            positive, state, snapshot = word > 0, word ^ 1, torch.zeros_like(word)
            for iteration in range(steps):
                earlier = torch.where(valid & positive, state, -17)
                state = torch.where(valid & ~positive, word ^ (iteration + 2) ^ earlier, state)
                observed = torch.where(valid, state, 0)
                if kind == 0:
                    mixed = observed.sum()
                else:
                    values, offsets = observed.tolist(), torch.where(valid, word, 0).tolist()
                    scale, offset = values[0], offsets[0]
                    for a, b in zip(values[1:], offsets[1:], strict=True):
                        scale, offset = signed(scale * a), signed(offset * a + b)
                    mixed = torch.tensor(signed(scale ^ offset), dtype=torch.int64)
                snapshot = snapshot ^ observed ^ mixed
                state = torch.where(valid & positive, observed ^ 8, state)
            if kind == 0 and steps <= 0:
                state = torch.where(valid & ~positive, word ^ 16, state)
            results.append(state[:length])
            snapshots.append(snapshot[:length])
        return torch.cat(results), torch.cat(snapshots)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([(1 << 54) + 1], dtype=torch.int64), requires_grad=False)

        def forward(self, x):
            n = x.shape[0]
            first, earlier = transform(x, self.bias, n, n - 2, 0)
            shifted = first + self.bias
            second, final = transform(shifted, self.bias, n, n - 1, 1)
            return second, first, earlier, final, x, shifted, self.bias

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        model, x = Model().eval(), torch.arange(17, dtype=torch.int64) + (1 << 55)
        x[1::2] *= -1
        exported = torch.export.export(model, (x,), dynamic_shapes=({0: torch.export.Dim("n", min=2, max=31)},))
        converted = m2m.convert(exported, (x,), backend="fx_importer", capture_trace=True,
                                weights_path=str(out / "weights.safetensors"))
        if not converted.ok:
            raise ValueError(converted.diagnostics)
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
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(capture(args.out))
