#!/usr/bin/env python3
"""Capture learned state, dependent early returns and every original result."""

import argparse
import sys
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t2m.examples.cfg_kernels import early_return_words
from t2m.examples.cfg_reference import update


def capture(out):
    out = Path(out).resolve()
    qualname = "t2m_demo::cfg_state"

    @triton_op(qualname, mutates_args={})
    def cfg_state(
        x: torch.Tensor, state: torch.Tensor, n: int, steps: int, mode: int
    ) -> torch.Tensor:
        result = state.contiguous().clone()
        if n:
            wrap_triton(early_return_words)[((n + 7) // 8,)](
                x, result, n, steps, mode, B=8
            )
        return result

    @cfg_state.register_kernel("cpu")
    def reference(x, state, n, steps, mode):
        return torch.from_numpy(
            update(x.detach().numpy(), state.detach().numpy(), n, steps, mode)
        )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            words = [0, 1, -1, -(1 << 63), (1 << 63) - 1, (1 << 53) + 1]
            self.weight = torch.nn.Parameter(
                torch.tensor((words * 4)[:19], dtype=torch.int64), requires_grad=False
            )
            self.bias = torch.nn.Parameter(
                torch.arange(19, dtype=torch.int64) % 7 - 3, requires_grad=False
            )

        def forward(self, x):
            original = x ^ self.weight
            first = cfg_state(x, original, 19, 3, 0)
            second = cfg_state(x, first, 19, 2, 1)
            empty = cfg_state(x[:0], original[:0], 0, 1, 1)
            return second + self.bias, first, empty, original

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (torch.arange(19, dtype=torch.int64) * 7 - 29,)
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
    print(capture(parser.parse_args().out))
