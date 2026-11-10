#!/usr/bin/env python3
"""Real model2MLIR capture with a functional atomic-update call."""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.library import triton_op, wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec

KINDS = ("add", "cas", "exch", "min", "max", "fadd", "fadd32", "word-cas16", "word-cas32", "word-cas64")


def capture(out, kind="add"):
    if kind not in KINDS:
        raise ValueError(f"atomic model kind must be one of {KINDS}")
    word_width = int(kind.removeprefix("word-cas")) if kind.startswith("word-cas") else None
    dtype = getattr(torch, f"float{word_width}") if word_width else torch.float64 if kind == "fadd" else torch.float32 if kind in ("exch", "min", "max", "fadd32") else torch.int32
    value_dtype = getattr(torch, f"int{word_width}") if word_width else dtype
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
        old = torch.empty_like(x, dtype=dtype)
        if word_width:
            wrap_triton(kernels.word_cas)[(2,)](updated, index, x, old, B=8, WIDTH=word_width, SCALAR=False)
        elif kind == "cas":
            wrap_triton(kernels.indexed_cas)[(2,)](updated, index, x, old, B=8,
                SCALAR=False, SEM="relaxed", SCOPE="gpu")
        else:
            wrap_triton(kernels.indexed_atomic)[(2,)](x, index, updated, old, n=13, m=17, B=8,
                OP="add" if kind in ("fadd", "fadd32") else kind, SCALAR=False, MASK=True, FRESH=False, SEM="relaxed", SCOPE="gpu")
        return updated, old

    @atomic_update.register_kernel("cpu")
    def reference(x, index, state):
        updated, old = state.clone(), torch.zeros_like(x, dtype=dtype)
        if word_width:
            storage, previous = updated.view(value_dtype), old.view(value_dtype)
            for lane in range(x.numel()):
                previous[lane] = storage[lane]
                if storage[lane] == index[lane]:
                    storage[lane] = x[lane]
            return updated, old
        for lane in range(x.numel()):
            if kind == "cas":
                old[lane] = updated[lane]
                if updated[lane] == index[lane]:
                    updated[lane] = x[lane]
            elif lane % 3 != 1:
                target = int(index[lane])
                old[lane] = updated[target]
                if kind == "exch":
                    updated[target] = x[lane]
                elif kind == "min":
                    updated[target] = torch.minimum(updated[target], x[lane])
                elif kind == "max":
                    updated[target] = torch.maximum(updated[target], x[lane])
                else:
                    updated[target] += x[lane]
        return updated, old

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state = torch.nn.Parameter(torch.arange(17, dtype=dtype) + 50, requires_grad=False)
            index = self.state.detach().view(value_dtype)[:16].clone() if word_width else torch.arange(16 if kind == "cas" else 13, dtype=torch.int32)
            if word_width:
                index[1::3] += 1
            elif kind == "cas":
                index += 50
                index[1::3] += 1
            else:
                index[torch.arange(13) % 3 == 1] = -1000
            self.register_buffer("index", index)

        def forward(self, x):
            return atomic_update(x, self.index, self.state)

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model, values = Model().eval(), (torch.arange(16 if kind == "cas" or word_width else 13, dtype=value_dtype) - 5,)
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
    parser.add_argument("--kind", choices=KINDS, default="add")
    arguments = parser.parse_args()
    print(capture(arguments.out, arguments.kind))
