#!/usr/bin/env python3
"""Capture genuine dynamic transpose/slice roots and complete state snapshots."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.fx.experimental import _config as export_config
from torch.library import triton_op, wrap_triton


def capture(out, dtype="i64"):
    source = Path(__file__).resolve().parents[1] / "t2m/examples/model_view_kernels.py"
    loader = importlib.util.spec_from_file_location("dynamic_source_view_kernels", source)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    storage = {"i64": torch.int64, "f32": torch.float32}[dtype]
    qualname = "t2m_demo::dynamic_source_view_update"

    @triton_op(qualname, mutates_args={})
    def update(owner: torch.Tensor, view: torch.Tensor, root_rows: int, root_cols: int,
               rows: int, cols: int, s0: int, s1: int, origin: int,
               steps: int, bias: int) -> tuple[torch.Tensor, torch.Tensor]:
        result = owner.contiguous().clone()
        selected = result.as_strided((rows, cols), (s0, s1), origin)
        values = torch.empty((rows, cols), dtype=owner.dtype, device=owner.device)
        wrap_triton(kernels.update_source_view)[((rows + 7) // 8, (cols + 3) // 4)](
            result, selected, values, rows, cols, s0, s1, steps, bias, BM=8, BN=4
        )
        return result, values

    @update.register_kernel("cpu")
    def reference(owner, view, root_rows, root_cols, rows, cols, s0, s1, origin, steps, bias):
        if (not owner.is_contiguous() or owner.storage_offset() != 0
                or tuple(owner.shape) != (root_rows, root_cols)
                or tuple(view.shape) != (rows, cols) or tuple(view.stride()) != (s0, s1)
                or view.storage_offset() != origin
                or owner.untyped_storage()._cdata != view.untyped_storage()._cdata):
            raise ValueError("dynamic source view differs from its actual owning storage relation")
        result = owner.clone()
        selected = result.as_strided((rows, cols), (s0, s1), origin)
        for _ in range(steps):
            selected.add_(bias)
        return result, selected.clone()

    @update.register_fake
    def fake(owner, view, root_rows, root_cols, rows, cols, s0, s1, origin, steps, bias):
        # The actual wrapper clones the owner and allocates the declared view
        # output. Its physical as_strided metadata need not specialize export.
        return torch.empty_like(owner), torch.empty((rows, cols), dtype=owner.dtype, device=owner.device)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor((1 << 53) + 1 if dtype == "i64" else -3,
                                                          dtype=storage), requires_grad=False)
            self.bias = torch.nn.Parameter(torch.tensor(-1, dtype=storage), requires_grad=False)

        def forward(self, x):
            initial = x + self.weight
            root_rows, root_cols = x.shape
            first_view = initial.t()[1::2, :]
            first, observed = update(initial, first_view, root_rows, root_cols,
                first_view.shape[0], first_view.shape[1], 2, root_cols, 1, 2, 5)
            next_view = first[::2, 2::3]
            second, later = update(first, next_view, root_rows, root_cols,
                next_view.shape[0], next_view.shape[1], 2 * root_cols, 3, 2, 1, -3)
            empty_view = first.t()[:0, :]
            empty, none = update(first, empty_view, root_rows, root_cols,
                0, root_rows, 1, root_cols, 0, 3, 2)
            full = second + self.bias
            return full, first, observed, later, empty, none, first_view, initial, x

    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = ((torch.arange(15).reshape(3, 5) % 7).to(storage),)
        # Torch's supported experimental mode keeps unknown contiguity metadata
        # conservative instead of specializing derived dimensions to zero/one.
        # It does not suppress the declared source range or value assertions.
        with export_config.patch(backed_size_oblivious=True):
            exported = torch.export.export(model, inputs, dynamic_shapes={"x": {
                0: torch.export.Dim("rows", min=2, max=9),
                1: torch.export.Dim("cols", min=2, max=17),
            }})
        out = Path(out).resolve()
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True,
                             weights_path=str(out / "weights.safetensors"))
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, inputs, out, capture_trace=True, capture_regions=False,
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
    parser.add_argument("--dtype", choices=("i64", "f32"), default="i64")
    arguments = parser.parse_args()
    print(capture(arguments.out, arguments.dtype))
