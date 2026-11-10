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


def dynamic_reference(a, b, sa, sb, initial, m, n, k, packed_k, groups):
    if packed_k != (k + 1) // 2 or groups != (k + 31) // 32:
        raise ValueError("packed/group extent differs from logical K")
    result = initial.clone()
    for step in range(groups):
        left = torch.zeros((m, 16), dtype=a.dtype)
        right = torch.zeros((16, n), dtype=b.dtype)
        start, stop = step * 16, min((step + 1) * 16, packed_k)
        left[:, : stop - start] = a[:, start:stop]
        right[: stop - start, :] = b[start:stop, :]
        if k % 2 and stop == packed_k:
            left[:, stop - start - 1] &= 15
            right[stop - start - 1, :] &= 15
        result = scaled_reference(
            left,
            right,
            sa[:, step : step + 1],
            sb[:, step : step + 1],
            result,
            True,
            True,
        )
    return result


def capture(out, *, dynamic=False):
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

    if dynamic:
        qualname = "t2m_demo::scaled_dot_dynamic"

        @triton_op(qualname, mutates_args={})
        def dynamic_dot(
            a: torch.Tensor,
            b: torch.Tensor,
            sa: torch.Tensor,
            sb: torch.Tensor,
            initial: torch.Tensor,
            m: int,
            n: int,
            k: int,
            packed_k: int,
            groups: int,
        ) -> torch.Tensor:
            out = torch.empty_like(initial)
            wrap_triton(kernels.scaled_dot_dynamic)[((m + 15) // 16, (n + 15) // 16)](
                a, b, sa, sb, initial, out, m, n, k, packed_k, groups, BM=16, BN=16
            )
            return out

        dynamic_dot.register_kernel("cpu")(dynamic_reference)

    m, n, k = (17, 19, 37) if dynamic else (16, 16, 64)
    packed_k, groups = (k + 1) // 2, (k + 31) // 32
    bshape = (packed_k, n) if dynamic else (k, n // 2)
    ashape = (m, packed_k) if dynamic else (m // 2, k)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rhs = torch.nn.Parameter(
                (255 - torch.arange(bshape[0] * bshape[1], dtype=torch.int32))
                .to(torch.int8)
                .reshape(bshape),
                requires_grad=False,
            )
            self.initial = torch.nn.Parameter(
                (torch.arange(m * n, dtype=torch.float32) % 11 - 5).reshape(m, n) * 0.5
            )
            self.initial.data[0, :2] = torch.tensor(
                [2**24, -(2**24)], dtype=torch.float32
            )
            self.bias = torch.nn.Parameter(
                (torch.arange(m * n, dtype=torch.float32) % 5 - 2).reshape(m, n) * 0.25
            )
            self.register_buffer(
                "lhs_scale",
                torch.tensor([126, 128, 127, 129], dtype=torch.int16)
                .to(torch.int8)
                .repeat((m * groups + 3) // 4)[: m * groups]
                .reshape(m, groups),
            )
            self.register_buffer(
                "rhs_scale",
                torch.tensor([128, 126, 127, 125], dtype=torch.int16)
                .to(torch.int8)
                .repeat((n * groups + 3) // 4)[: n * groups]
                .reshape(n, groups),
            )

        def forward(self, x):
            if dynamic:
                output = dynamic_dot(
                    x,
                    self.rhs,
                    self.lhs_scale,
                    self.rhs_scale,
                    self.initial,
                    m,
                    n,
                    k,
                    packed_k,
                    groups,
                )
            else:
                output = scaled_dot(
                    x, self.rhs, self.lhs_scale, self.rhs_scale, self.initial
                )
            return output + self.bias, self.initial + self.bias

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        values = (
            torch.arange(ashape[0] * ashape[1], dtype=torch.int32)
            .to(torch.int8)
            .reshape(ashape),
        )
        if dynamic and k % 2:
            values[0][:, -1] |= -16
            model.rhs.data[-1, :] |= -16
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
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()
    print(capture(args.out, dynamic=args.dynamic))
