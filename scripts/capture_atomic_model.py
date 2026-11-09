#!/usr/bin/env python3
"""Real model2MLIR capture with a functional integer atomic-update call."""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.library import triton_op, wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


def capture(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = Path(__file__).resolve().parents[1] / "t2m/examples/atomic_kernels.py"
    spec = importlib.util.spec_from_file_location("t2m_atomic_kernels", source)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::atomic_update"

    @triton_op(qualname, mutates_args={})
    def atomic_update(x: torch.Tensor, index: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        updated = state.clone()
        old = torch.empty_like(x)
        wrap_triton(kernels.indexed_atomic)[(2,)](x, index, updated, old, n=13, m=17, B=8,
            OP="add", SCALAR=False, MASK=True, FRESH=False, SEM="relaxed", SCOPE="gpu")
        return updated, old

    @atomic_update.register_kernel("cpu")
    def reference(x, index, state):
        updated, old = state.clone(), torch.zeros_like(x)
        for lane in range(x.numel()):
            if lane % 3 != 1:
                target = int(index[lane])
                old[lane] = updated[target]
                updated[target] += x[lane]
        return updated, old

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state = torch.nn.Parameter(torch.arange(17, dtype=torch.int32) + 50, requires_grad=False)
            index = torch.arange(13, dtype=torch.int32)
            index[torch.arange(13) % 3 == 1] = -1000
            self.register_buffer("index", index)

        def forward(self, x):
            return atomic_update(x, self.index, self.state)

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model, values = Model().eval(), (torch.arange(13, dtype=torch.int32) - 5,)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(model, values, backend="fx_importer", weights_path=str(out / "weights.safetensors"), capture_trace=True)
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, values, out, capture_regions=False, capture_trace=True,
                     source_path=Path(__file__), conversion_result=result)
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
