#!/usr/bin/env python3
"""Capture dependent local-tensor kernels, model parameters and every result."""

import argparse
import sys
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2m.examples.tensor_region_kernels import tensor_region_bitsets


def capture(out, dtype="i64"):
    out = Path(out).resolve()
    if dtype not in ("i32", "i64"):
        raise ValueError("the model example requires i32 or i64 source words")
    storage = torch.int32 if dtype == "i32" else torch.int64
    name = "t2m_demo::tensor_regions"

    @triton_op(name, mutates_args={})
    def bitsets(x: torch.Tensor, n: int, kind: int) -> torch.Tensor:
        result = torch.empty_like(x)
        if n:
            wrap_triton(tensor_region_bitsets)[((n + 7) // 8,)](
                x, result, n, B=8, KIND=("map", "reduce", "scan")[kind],
                WHILE=kind == 2, REVERSE=kind == 2,
            )
        return result

    @bitsets.register_kernel("cpu")
    def reference(x, n, kind):
        # Independent source-level math; compiler helpers and generated MLIR
        # supply neither this oracle nor the captured enclosing-model goldens.
        result = x.clone()
        for start in range(0, n, 8):
            stop = min(start + 8, n)
            if kind == 1:
                value = -1
                for word in x[start:stop]:
                    value &= int(word)
                result[start:stop] = value
            elif kind == 2:
                value = -1
                for index in reversed(range(start, stop)):
                    value &= int(x[index])
                    result[index] = value
        return result

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            high = 1 << (29 if dtype == "i32" else 60)
            values = [high + index * 7 if index % 2 else -high + index * 11
                      for index in range(19)]
            self.weight = torch.nn.Parameter(torch.tensor(values, dtype=storage), requires_grad=False)
            self.bias = torch.nn.Parameter(torch.arange(19, dtype=storage) - 7, requires_grad=False)

        def forward(self, x):
            original = x + self.weight
            first = bitsets(original, 19, 0)
            reduced = bitsets(first, 19, 1)
            scanned = bitsets(original, 19, 2)
            dependent = bitsets(scanned + self.bias, 19, 1)
            empty = bitsets(x[:0], 0, 0)
            return dependent, reduced, scanned, first, empty, original, first + self.bias

    previous = KERNEL_REGISTRY.get(name)
    try:
        KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (torch.arange(19, dtype=storage) * 37 - 191,)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
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
    parser.add_argument("--dtype", choices=("i32", "i64"), default="i64")
    args = parser.parse_args()
    print(capture(args.out, args.dtype))
