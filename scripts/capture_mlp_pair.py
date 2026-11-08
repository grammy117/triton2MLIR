#!/usr/bin/env python3
"""Capture model2MLIR's existing plain/Triton MLP pair, with eager CPU goldens.

This uses the workloads' original models, initialization and Triton op. The
registered CPU reference evaluates the same math without a Triton target backend.
"""

import argparse
import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

import m2m
from m2m.capture.bundle import write_bundle


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    plain = load(root / "workloads/mlp_basic/loader.py", "plain_mlp_fixture")
    fused = load(root / "workloads/mlp_basic_triton/loader.py", "triton_mlp_fixture")
    operation = sys.modules["kernels"].fused_mlp2

    @operation.register_kernel("cpu")
    def reference(x, w1, b1, w2, b2):
        return F.relu(F.linear(F.relu(F.linear(x, w1, b1)), w2, b2))

    torch.set_num_threads(1)
    for name, loader in (("plain", plain), ("triton", fused)):
        directory = out / name
        directory.mkdir(parents=True)
        model, inputs = loader.get_model_and_inputs()
        result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
                             weights_path=str(directory / "weights.safetensors"))
        if not result.ok:
            raise ValueError(result.diagnostics)
        write_bundle(model, inputs, directory, capture_regions=False, capture_trace=True,
                     source_path=Path(loader.__file__), conversion_result=result)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(capture(args.model2mlir_root, args.out))
