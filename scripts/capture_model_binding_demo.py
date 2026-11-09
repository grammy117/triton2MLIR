#!/usr/bin/env python3
"""Capture a real model2MLIR Triton-op model using its existing environment.

The registered CPU implementation supplies the independent PyTorch golden.
No Triton CPU backend or GPU kernel is compiled or executed. The Triton body is
captured separately by triton2MLIR using its pinned frontend environment.
"""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.library import triton_op, wrap_triton

from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
import m2m


def capture(out, *, ragged=False, block_pointers=False):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    # Keep the independently pinned m2m environment from importing the t2m
    # compiler package (and its different xDSL/Triton version requirements).
    path = Path(__file__).resolve().parents[1] / "t2m/examples/tensor_kernels.py"
    spec = importlib.util.spec_from_file_location("t2m_demo_kernels", path)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    loop_matmul = kernels.loop_matmul
    ragged_matmul = kernels.ragged_matmul
    blockptr_matmul = kernels.blockptr_matmul
    ragged = ragged or block_pointers

    @triton_op("t2m_demo::gemm", mutates_args={})
    def gemm(x: torch.Tensor, y: torch.Tensor, alpha: int = 1) -> torch.Tensor:
        m, k = x.shape
        n = y.shape[1]
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        if block_pointers:
            wrap_triton(blockptr_matmul)[((m + 15) // 16, (n + 15) // 16)](
                x, y, out, M=m, N=n, K=k, BM=16, BN=16, BK=16)
        elif ragged:
            wrap_triton(ragged_matmul)[((m + 15) // 16, (n + 15) // 16)](
                x, y, out, M=m, N=n, K=k, BM=16, BN=16, BK=16, TRANSPOSED=False)
        else:
            wrap_triton(loop_matmul)[(m // 16, n // 16)](
                x, y, out, N=n, K=k, BM=16, BN=16, BK=16)
        return out * alpha

    @gemm.register_kernel("cpu")
    def reference(x, y, alpha=1):
        return (x @ y) * alpha

    rows, inner, columns = (35, 37, 21) if ragged else (32, 32, 32)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(inner, columns) / 4)

        def forward(self, x):
            value = gemm(x, self.weight, alpha=1)
            return value + 0.25 if ragged else value + x * 0.25

    previous = KERNEL_REGISTRY.get("t2m_demo::gemm")
    KERNEL_REGISTRY["t2m_demo::gemm"] = KernelSpec("t2m_demo::gemm")
    try:
        torch.set_num_threads(1)
        torch.manual_seed(24)
        model, inputs = Model().eval(), (torch.randn(rows, inner),)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(model, inputs, backend="fx_importer",
                             weights_path=str(out / "weights.safetensors"), capture_trace=True)
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, inputs, out, capture_regions=False, capture_trace=True,
                     source_path=Path(__file__), conversion_result=result)
    finally:
        if previous is None:
            KERNEL_REGISTRY.pop("t2m_demo::gemm", None)
        else:
            KERNEL_REGISTRY["t2m_demo::gemm"] = previous
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ragged", action="store_true", help="capture a 35x37 by 37x21 tiled GEMM")
    parser.add_argument("--block-pointers", action="store_true", help="capture ragged GEMM with block pointers and advance")
    args = parser.parse_args()
    print(capture(args.out, ragged=args.ragged, block_pointers=args.block_pointers))
