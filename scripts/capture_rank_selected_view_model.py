#!/usr/bin/env python3
"""Capture rank-reducing row/column/scalar views with complete owning state."""

import argparse
import importlib.util
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec
from torch.fx.experimental import _config
from torch.library import triton_op, wrap_triton


def capture(out, dtype="i64"):
    path = Path(__file__).resolve().parents[1] / "t2m/examples/model_view_kernels.py"
    loader = importlib.util.spec_from_file_location("rank_selected_view_kernel", path)
    kernels = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(kernels)
    storage = {"i64": torch.int64, "f32": torch.float32}[dtype]
    names = ("t2m_demo::rank_selected_view_update", "t2m_demo::rank_selected_scalar_update")

    def operation(qualname, scalar):
        @triton_op(qualname, mutates_args={})
        def update(owner: torch.Tensor, view: torch.Tensor, root_rows: int, root_cols: int,
                   rows: int, cols: int, s0: int, s1: int, origin: int,
                   steps: int, bias: int) -> tuple[torch.Tensor, torch.Tensor]:
            result = owner.contiguous().clone()
            selected = result.as_strided((), (), origin) if scalar else result.as_strided((rows, cols), (s0, s1), origin)
            output = torch.empty(() if scalar else (rows, cols), dtype=owner.dtype, device=owner.device)
            wrap_triton(kernels.update_source_view)[((rows + 7) // 8, (cols + 3) // 4)](
                result, selected, output, rows, cols, s0, s1, steps, bias, BM=8, BN=4)
            return result, output

        @update.register_kernel("cpu")
        def reference(owner, view, root_rows, root_cols, rows, cols, s0, s1, origin, steps, bias):
            shape, strides = ((), ()) if scalar else ((rows, cols), (s0, s1))
            if (not owner.is_contiguous() or owner.storage_offset() != 0
                    or tuple(owner.shape) != (1, root_rows, root_cols) or tuple(view.shape) != shape
                    or tuple(view.stride()) != strides or view.storage_offset() != origin
                    or owner.untyped_storage()._cdata != view.untyped_storage()._cdata
                    or scalar and (rows != 1 or cols != 1)):
                raise ValueError("selected view differs from its original owning tensor relationship")
            result = owner.clone()
            selected = result.as_strided(shape, strides, origin)
            for _ in range(steps):
                selected.add_(bias)
            return result, selected.clone()

        @update.register_fake
        def fake(owner, view, root_rows, root_cols, rows, cols, s0, s1, origin, steps, bias):
            return torch.empty_like(owner), torch.empty(() if scalar else (rows, cols), dtype=owner.dtype, device=owner.device)

        return update

    update, update_scalar = operation(names[0], False), operation(names[1], True)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor((1 << 53) + 1 if dtype == "i64" else -3,
                dtype=storage), requires_grad=False)
            self.bias = torch.nn.Parameter(torch.tensor(-1, dtype=storage), requires_grad=False)

        def forward(self, x):
            initial = x + self.weight
            _unit, root_rows, root_cols = x.shape
            first_view = initial[:, 1, :]
            first, observed = update(initial, first_view, root_rows, root_cols,
                1, root_cols, root_rows * root_cols, 1, root_cols, 2, 5)
            column_view = first[:, :, -1]
            second, later = update(first, column_view, root_rows, root_cols,
                1, root_rows, root_rows * root_cols, root_cols, root_cols - 1, 1, -3)
            scalar_view = second[0, 1, -1]
            third, number = update_scalar(second, scalar_view, root_rows, root_cols,
                1, 1, 1, 1, 2 * root_cols - 1, 3, 2)
            empty_view = third[:, 1, :0]
            empty, none = update(third, empty_view, root_rows, root_cols,
                1, 0, root_rows * root_cols, 1, root_cols, 2, 4)
            full = third + self.bias
            return full, first, observed, later, third, number, empty, none, first_view, column_view, scalar_view, initial, x

    previous = {name: KERNEL_REGISTRY.get(name) for name in names}
    try:
        for name in names:
            KERNEL_REGISTRY[name] = KernelSpec(name)
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = ((torch.arange(15).reshape(1, 3, 5) % 7).to(storage),)
        with _config.patch(backed_size_oblivious=True):
            exported = torch.export.export(model, inputs, dynamic_shapes={"x": {
                1: torch.export.Dim("rows", min=2, max=9), 2: torch.export.Dim("cols", min=2, max=17)}})
        out = Path(out).resolve()
        out.mkdir(parents=True, exist_ok=False)
        result = m2m.convert(exported, inputs, backend="fx_importer", capture_trace=True,
            weights_path=str(out / "weights.safetensors"))
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, inputs, out, capture_trace=True, capture_regions=False,
            source_path=Path(__file__), conversion_result=result)
    finally:
        for name, old in previous.items():
            if old is None:
                KERNEL_REGISTRY.pop(name, None)
            else:
                KERNEL_REGISTRY[name] = old
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("i64", "f32"), default="i64")
    args = parser.parse_args()
    print(capture(args.out, args.dtype))
