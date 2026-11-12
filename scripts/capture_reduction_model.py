#!/usr/bin/env python3
"""Capture learned state, coupled reductions and every original model result."""

import argparse
import sys
from pathlib import Path

import m2m
import numpy as np
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2m.examples.reduction_kernels import reduce_recurrence
from t2m.examples.reduction_reference import recurrence


def capture(out, kind, axis):
    if kind not in ("affine", "argmin"):
        raise ValueError("whole-model selection must be affine or argmin")
    qualname = "t2m_demo::reduction_state"

    @triton_op(qualname, mutates_args={})
    def reduction_state(
        x: torch.Tensor, y: torch.Tensor, n: int, steps: int, kind: str, axis: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = torch.empty_like(x), torch.empty_like(y)
        if n:
            wrap_triton(reduce_recurrence)[((n + 7) // 8,)](
                x, y, first, second, n, steps, B=8, KIND=kind, AXIS=axis
            )
        return first, second

    @reduction_state.register_kernel("cpu")
    def reference(x, y, n, steps, kind, axis):
        outputs = recurrence(
            x.detach().numpy(), y.detach().numpy(), n, steps, kind, axis
        )
        return tuple(torch.from_numpy(value) for value in outputs)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "affine":
                self.weight = torch.nn.Parameter(
                    torch.arange(19, dtype=torch.int64) + (1 << 62) + 1,
                    requires_grad=False,
                )
                self.bias = torch.nn.Parameter(
                    torch.arange(19, dtype=torch.int64) * -7 + (1 << 62) + 3,
                    requires_grad=False,
                )
            else:
                words = np.resize(
                    np.asarray(
                        [
                            0x80000000,
                            0x7FC00001,
                            0x3F800000,
                            0xBF800000,
                            0xFFC00013,
                            0xFF800000,
                            0,
                            0xBF800000,
                        ],
                        np.uint32,
                    ),
                    19,
                )
                self.weight = torch.nn.Parameter(
                    torch.from_numpy(words.view(np.float32)), requires_grad=False
                )
                self.fallback = torch.nn.Parameter(
                    (torch.arange(19) % 5 - 2).to(torch.float32), requires_grad=False
                )
                self.bias = torch.nn.Parameter(
                    (torch.arange(19, dtype=torch.int32) % 7 - 3) * 5,
                    requires_grad=False,
                )

        def forward(self, x):
            original = (
                x ^ self.weight
                if kind == "affine"
                else torch.where(x > 0, self.weight, self.fallback)
            )
            first_a, first_b = reduction_state(original, self.bias, 19, 1, kind, axis)
            second_a, second_b = reduction_state(first_a, first_b, 19, 2, kind, axis)
            empty_a, empty_b = reduction_state(
                original[:0], self.bias[:0], 0, 3, kind, axis
            )
            return (
                second_a,
                second_b,
                first_a,
                first_b,
                empty_a,
                empty_b,
                original,
                self.bias,
            )

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    out = Path(out).resolve()
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (
            (torch.arange(19, dtype=torch.int64) * 3 - 17,)
            if kind == "affine"
            else (torch.arange(19, dtype=torch.float32) % 5 - 1,)
        )
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(
            model,
            inputs,
            backend="fx_importer",
            weights_path=str(out / "weights.safetensors"),
            capture_trace=True,
        )
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(
            model,
            inputs,
            out,
            capture_regions=False,
            capture_trace=True,
            source_path=Path(__file__),
            conversion_result=result,
        )
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop(qualname, None)
        else:
            KERNEL_REGISTRY[qualname] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=("affine", "argmin"), required=True)
    parser.add_argument("--axis", choices=(0, 1), type=int, default=1)
    args = parser.parse_args()
    print(capture(args.out, args.kind, args.axis))
