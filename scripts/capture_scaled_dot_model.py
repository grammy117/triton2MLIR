#!/usr/bin/env python3
"""Real model2MLIR scaled-dot capture with weights and independent CPU math."""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.library import triton_op, wrap_triton

import m2m
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec


def unpack(words, axis):
    words = words.to(torch.int16) & 255
    codes = torch.stack((words & 15, words >> 4), dim=axis + 1).flatten(axis, axis + 1)
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64
    )[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -magnitude, magnitude)


def scaled_reference(a, b, sa, sb, initial, lhs_k_pack, rhs_k_pack):
    left = unpack(a, 1 if lhs_k_pack else 0)
    right = unpack(b, 0 if rhs_k_pack else 1)

    def scale(words):
        codes = words.to(torch.int16) & 255
        values = torch.ldexp(torch.ones_like(codes, dtype=torch.float64), codes - 127)
        return torch.where(codes == 255, torch.nan, values)

    left = left * scale(sa).repeat_interleave(32, dim=1)
    right = right * scale(sb).repeat_interleave(32, dim=1).T
    accumulator = initial.to(torch.float64)
    for k in range(left.shape[1]):
        accumulator = accumulator + left[:, k, None] * right[k, None, :]
    return accumulator.to(torch.float32)


def capture(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    path = Path(__file__).resolve().parents[1] / "t2m/examples/scaled_dot_kernels.py"
    spec = importlib.util.spec_from_file_location("scaled_model_kernels", path)
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    qualname = "t2m_demo::scaled_dot"

    @triton_op(qualname, mutates_args={})
    def scaled_dot(
        a: torch.Tensor,
        b: torch.Tensor,
        sa: torch.Tensor,
        sb: torch.Tensor,
        initial: torch.Tensor,
        lhs_k_pack: bool = False,
        rhs_k_pack: bool = False,
    ) -> torch.Tensor:
        out = torch.empty_like(initial)
        wrap_triton(kernels.scaled_dot_tile)[(1,)](
            a,
            b,
            sa,
            sb,
            initial,
            out,
            M=16,
            N=16,
            K=64,
            A_FORMAT="e2m1",
            B_FORMAT="e2m1",
            LHS_K_PACK=lhs_k_pack,
            RHS_K_PACK=rhs_k_pack,
            HAS_A_SCALE=True,
            HAS_B_SCALE=True,
            GROUP=32,
        )
        return out

    @scaled_dot.register_kernel("cpu")
    def reference(a, b, sa, sb, initial, lhs_k_pack=False, rhs_k_pack=False):
        return scaled_reference(a, b, sa, sb, initial, lhs_k_pack, rhs_k_pack)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rhs = torch.nn.Parameter(
                (255 - torch.arange(512, dtype=torch.int32))
                .to(torch.int8)
                .reshape(64, 8),
                requires_grad=False,
            )
            self.initial = torch.nn.Parameter(
                (torch.arange(256, dtype=torch.float32) % 11 - 5).reshape(16, 16) * 0.5
            )
            self.initial.data[0, :2] = torch.tensor(
                [2**24, -(2**24)], dtype=torch.float32
            )
            self.bias = torch.nn.Parameter(
                (torch.arange(256, dtype=torch.float32) % 5 - 2).reshape(16, 16) * 0.25
            )
            self.register_buffer(
                "lhs_scale",
                torch.tensor([126, 128, 127, 129], dtype=torch.int16)
                .to(torch.int8)
                .repeat(8)
                .reshape(16, 2),
            )
            self.register_buffer(
                "rhs_scale",
                torch.tensor([128, 126, 127, 125], dtype=torch.int16)
                .to(torch.int8)
                .repeat(8)
                .reshape(16, 2),
            )

        def forward(self, x):
            output = scaled_dot(
                x, self.rhs, self.lhs_scale, self.rhs_scale, self.initial
            )
            return output + self.bias, self.initial + self.bias

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        values = (torch.arange(512, dtype=torch.int32).to(torch.int8).reshape(8, 64),)
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(
            model,
            values,
            backend="fx_importer",
            weights_path=str(out / "weights.safetensors"),
            capture_trace=True,
        )
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(
            model,
            values,
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
    args = parser.parse_args()
    print(capture(args.out))
