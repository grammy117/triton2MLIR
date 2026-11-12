#!/usr/bin/env python3
"""Capture learned half state, dependent atomic calls and original snapshots."""

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
from t2m.examples.float_cast_reference import add_word
from t2m.examples.half_atomic_kernels import half_atomic_add


def capture(out, dtype="f16"):
    out = Path(out).resolve()
    if dtype not in ("f16", "bf16"):
        raise ValueError("half model dtype must be f16 or bf16")
    carrier = torch.float16 if dtype == "f16" else torch.bfloat16
    qualname = "t2m_demo::half_atomic_state"

    @triton_op(qualname, mutates_args={})
    def update(
        x: torch.Tensor, index: torch.Tensor, state: torch.Tensor, n: int, m: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        updated, old = state.clone(), torch.empty_like(x, dtype=torch.int16)
        if n:
            wrap_triton(half_atomic_add)[((n + 7) // 8,)](
                x, index, updated, old, n, m, B=8, SCALAR=False
            )
        return updated, old

    @update.register_kernel("cpu")
    def reference(x, index, state, n, m):
        incoming = state.detach().view(torch.int16).numpy().view(np.uint16).copy()
        increments = x.detach().view(torch.int16).numpy().view(np.uint16)
        old = np.zeros(n, np.uint16)
        for lane in range(n):
            if lane % 3 != 1:
                destination = int(index[lane])
                old[lane] = incoming[destination]
                incoming[destination] = add_word(
                    int(incoming[destination]), int(increments[lane]), dtype
                )
        return torch.from_numpy(incoming.view(np.int16)).view(
            carrier
        ), torch.from_numpy(old.view(np.int16))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state = torch.nn.Parameter(
                ((torch.arange(37) % 13 - 6) / 8).to(carrier), requires_grad=False
            )
            self.bias = torch.nn.Parameter(
                ((torch.arange(37) % 5 - 2) / 32).to(carrier), requires_grad=False
            )
            index = torch.arange(31, dtype=torch.int32)
            index[index % 3 == 1] = -8191
            self.register_buffer("index", index)

        def forward(self, x):
            original = self.state + self.bias
            first, old_first = update(x, self.index, original, 31, 37)
            second, old_second = update(x, self.index, first, 31, 37)
            _, empty = update(x[:0], self.index[:0], original, 0, 37)
            return second + self.bias, first, old_first, old_second, empty, original

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        inputs = (((torch.arange(31) % 7 - 3) / 128).to(carrier),)
        model = Model().eval()
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
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    args = parser.parse_args()
    print(capture(args.out, args.dtype))
