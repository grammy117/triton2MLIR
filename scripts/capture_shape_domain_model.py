#!/usr/bin/env python3
"""Capture actual symbolic input domains around two channel reductions."""

import argparse
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle


def capture(out, mode="affine"):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x.sum(dim=0) + y.sum(dim=0)

    torch.set_num_threads(1)
    model = Model().eval()
    dimension = torch.export.Dim("n", min=0 if mode == "zero" else 2,
                                 **({} if mode == "unbounded" else {"max": 31}))
    x_rows = 13 if mode != "inverted" else 27
    y_rows = 27 if mode != "shared" else 13
    x = torch.arange(x_rows * 3, dtype=torch.float32).reshape(x_rows, 3) * .125
    y = torch.arange(y_rows * 3, dtype=torch.float32).reshape(y_rows, 3) * -.25
    program = torch.export.export(model, (x, y), dynamic_shapes={
        "x": {0: 2 * dimension + 1 if mode == "inverted" else dimension},
        "y": {0: dimension if mode == "shared" else 2 * dimension + 1}})
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = m2m.convert(program, (x, y), backend="fx_importer", capture_trace=True,
                         weights_path=str(out / "weights.safetensors"))
    if not result.ok:
        raise ValueError(result.diagnostics)
    from m2m.coverage import opaque_report
    if opaque_report(result.mlir_text):
        raise ValueError("shape model contains unresolved computation")
    write_bundle(model, (x, y), out, capture_trace=True, capture_regions=False,
                 conversion_result=result, source_path=Path(__file__))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("affine", "shared", "inverted", "zero", "unbounded"), default="affine")
    options = parser.parse_args()
    print(capture(options.out, options.mode))
