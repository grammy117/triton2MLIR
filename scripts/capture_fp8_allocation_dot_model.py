#!/usr/bin/env python3
"""Capture mixed-precision tiled model calls with owning accumulator state."""

import argparse
import importlib.util
import sys
from pathlib import Path

import m2m
import numpy as np
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.library import triton_op, wrap_triton
from triton.tools.tensor_descriptor import TensorDescriptor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t2m.examples.fp8_allocation_dot_reference import reference as dot_reference

MODES = {"bf16x3": 3, "bf16x6": 4, "tf32x3": 1}


def capture(out, *, precision="bf16x3", descriptors=False):
    out = Path(out).resolve()
    spec = importlib.util.spec_from_file_location(
        "fp8_allocation_model_kernel",
        ROOT / "t2m/examples/fp8_allocation_dot_kernels.py",
    )
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::fp8_allocation_dot"

    @triton_op(qualname, mutates_args={})
    def fp8_allocation_dot(
        x: torch.Tensor,
        weight: torch.Tensor,
        state: torch.Tensor,
        m: int,
        n: int,
        k: int,
        a_row: int,
        a_k: int,
        b_k: int,
        b_col: int,
        a_base: int,
        b_base: int,
        a_count: int,
        b_count: int,
        out_row: int,
        out_col: int,
        out_base: int,
        out_count: int,
        c_origin: int,
        precision_code: int,
        descriptors: bool,
    ) -> torch.Tensor:
        result = state.contiguous().clone()
        if not k:
            return result
        left, right = x.contiguous(), weight.contiguous()
        if descriptors:
            left = TensorDescriptor(
                left.view(-1)[a_base:], [m, k], [a_row, a_k], [16, 32]
            )
            right = TensorDescriptor(
                right.view(-1)[b_base:], [k, n], [b_k, b_col], [32, 16]
            )
        mode = next(name for name, code in MODES.items() if code == precision_code)
        flat = result.view(-1)
        wrap_triton(kernels.fp8_allocation_dot)[((m + 15) // 16, (n + 15) // 16)](
            left,
            right,
            flat[c_origin:],
            flat,
            m,
            n,
            k,
            a_row,
            a_k,
            b_k,
            b_col,
            a_base,
            b_base,
            a_count,
            b_count,
            out_row,
            out_col,
            out_base,
            out_count,
            BM=16,
            BN=16,
            BK=32,
            PRECISION=mode,
            DESCRIPTORS=descriptors,
        )
        return result

    @fp8_allocation_dot.register_kernel("cpu")
    def reference(
        x,
        weight,
        state,
        m,
        n,
        k,
        a_row,
        a_k,
        b_k,
        b_col,
        a_base,
        b_base,
        a_count,
        b_count,
        out_row,
        out_col,
        out_base,
        out_count,
        c_origin,
        precision_code,
        descriptors,
    ):
        if (
            x.numel() != a_count
            or weight.numel() != b_count
            or tuple(state.shape) != (m, n)
            or (out_row, out_col, out_base, out_count, c_origin) != (n, 1, 0, m * n, 0)
        ):
            raise ValueError(
                "reference requires the declared owning roots and output view"
            )
        mode = next(name for name, code in MODES.items() if code == precision_code)
        left = (
            np.ndarray(
                (m, k),
                np.float32,
                buffer=x.detach().numpy(),
                offset=a_base * 4,
                strides=(a_row * 4, a_k * 4),
            )
            if k
            else np.empty((m, 0), np.float32)
        )
        right = (
            np.ndarray(
                (k, n),
                np.float32,
                buffer=weight.detach().numpy(),
                offset=b_base * 4,
                strides=(b_k * 4, b_col * 4),
            )
            if k
            else np.empty((0, n), np.float32)
        )
        return torch.from_numpy(
            dot_reference(
                left,
                right,
                state.detach().numpy(),
                mode,
            )
        )

    m, n, k, a_stride, b_stride = 17, 5, 37, 40, 8

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                (
                    torch.arange(a_stride * b_stride).reshape(a_stride, b_stride) % 13
                    - 6
                ).float()
                * 0.08313,
                requires_grad=False,
            )
            initial = (torch.arange(m * n).reshape(m, n) % 7 - 3).float() * 0.31419
            initial[0, 0], initial[0, 1] = 2**24, -0.0
            self.initial = torch.nn.Parameter(initial, requires_grad=False)
            self.bias = torch.nn.Parameter(
                (torch.arange(n) % 5 - 2).float() * 0.031419, requires_grad=False
            )

        def forward(self, x):
            first = fp8_allocation_dot(
                x,
                self.weight,
                self.initial,
                m,
                n,
                k,
                a_stride,
                1,
                b_stride,
                1,
                0,
                0,
                m * a_stride,
                a_stride * b_stride,
                n,
                1,
                0,
                m * n,
                0,
                MODES[precision],
                descriptors,
            )
            second = fp8_allocation_dot(
                x,
                self.weight,
                first,
                m,
                n,
                k,
                a_stride,
                1,
                b_stride,
                1,
                0,
                0,
                m * a_stride,
                a_stride * b_stride,
                n,
                1,
                0,
                m * n,
                0,
                MODES[precision],
                descriptors,
            )
            empty_k = fp8_allocation_dot(
                x[:, :0],
                self.weight[:0],
                self.initial,
                m,
                n,
                0,
                0,
                1,
                b_stride,
                1,
                0,
                0,
                0,
                0,
                n,
                1,
                0,
                m * n,
                0,
                MODES[precision],
                descriptors,
            )
            return second + self.bias, first, empty_k, self.initial

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (
            (torch.arange(m * a_stride).reshape(m, a_stride) % 17 - 8).float()
            * 0.12731,
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
    parser.add_argument("--precision", choices=MODES, default="bf16x3")
    parser.add_argument("--descriptors", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, precision=args.precision, descriptors=args.descriptors))
