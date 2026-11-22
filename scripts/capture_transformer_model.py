#!/usr/bin/env python3
"""Capture ordinary PyTorch -> fused Triton transformer blocks automatically.

Run in the existing model2MLIR/Torch environment. The independent CPU reference
is only used to observe expected results. Fake launch discovery supplies source,
ABI and operand/result bindings without running or compiling a GPU kernel.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.library import triton_op, wrap_triton

REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kernels = load("_transformer_demo_source", REPO / "t2m/examples/transformer_kernels.py")
discovery = load("_transformer_launch_discovery", REPO / "t2m/torch_launch_capture.py")


def reference(x, wq, wk, wv, wo, w1, w2, g1, b1, g2, b2):
    norm = torch.nn.functional.layer_norm(x, (x.shape[-1],), g1, b1, 1e-5)
    query, key, value = norm @ wq, norm @ wk, norm @ wv
    probability = torch.softmax((query @ key.T) * (x.shape[-1] ** -0.5), dim=-1)
    residual = x + (probability @ value) @ wo
    norm2 = torch.nn.functional.layer_norm(residual, (x.shape[-1],), g2, b2, 1e-5)
    return residual + torch.relu(norm2 @ w1) @ w2


@triton_op("t2m_transformer::block", mutates_args={})
def fused(x: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor,
          wo: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, g1: torch.Tensor,
          b1: torch.Tensor, g2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    import triton

    out = torch.empty_like(x)
    wrap_triton(kernels.transformer_block)[(1,)](x, wq, wk, wv, wo, w1, w2, g1, b1, g2, b2, out,
        S=x.shape[0], D=x.shape[1], H=w1.shape[1], BS=triton.next_power_of_2(x.shape[0]), EPS=1e-5)
    return out


fused.register_kernel("cpu")(reference)


class TorchBlock(torch.nn.Module):
    """Full pre-norm encoder block, one attention head, ReLU FFN, no dropout."""

    def __init__(self, width=16, hidden=32):
        super().__init__()
        for name, shape in (("wq", (width, width)), ("wk", (width, width)),
                            ("wv", (width, width)), ("wo", (width, width)),
                            ("w1", (width, hidden)), ("w2", (hidden, width))):
            setattr(self, name, torch.nn.Parameter(torch.randn(shape) / (shape[0] ** 0.5)))
        for name in ("g1", "g2"):
            setattr(self, name, torch.nn.Parameter(torch.ones(width) + torch.randn(width) * 0.05))
        for name in ("b1", "b2"):
            setattr(self, name, torch.nn.Parameter(torch.randn(width) * 0.05))

    def arguments(self):
        return (self.wq, self.wk, self.wv, self.wo, self.w1, self.w2, self.g1, self.b1, self.g2, self.b2)

    def forward(self, x):
        return reference(x, *self.arguments())


class TritonBlock(TorchBlock):
    def forward(self, x):
        return fused(x, *self.arguments())


class TwoBlocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.regular = TorchBlock()
        self.triton = TritonBlock()

    def forward(self, x):
        first = self.regular(x)
        return first, self.triton(first)


def capture(out, *, tokens=16):
    import m2m
    from m2m.capture.bundle import write_bundle
    from m2m.coverage import opaque_report

    torch.set_num_threads(1)
    torch.manual_seed(20261121)
    model = TwoBlocks().eval()
    inputs = (torch.randn(tokens, 16),)
    launches = discovery.discover(model, inputs)
    if len(launches["calls"]) != 1:
        raise ValueError("two-block demo must discover exactly the second block's actual launch")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    parent = out / "parent"
    parent.mkdir()
    with discovery.register_discovered(launches):
        converted = m2m.convert(model, inputs, backend="fx_importer",
            weights_path=str(parent / "weights.safetensors"), capture_trace=True)
        if not converted.ok or opaque_report(converted.mlir_text):
            raise ValueError("ordinary transformer computation did not lower completely: " + str(converted.diagnostics))
        write_bundle(model, inputs, parent, capture_regions=False, capture_trace=True,
            source_path=Path(__file__), conversion_result=converted)
    (out / "launches.json").write_text(json.dumps(launches, indent=2, sort_keys=True) + "\n")
    references = out / "references"
    references.mkdir()
    with torch.no_grad():
        for ordinal, value in enumerate((inputs[0], torch.randn_like(inputs[0]) * 0.4 - 0.7,
                                         torch.zeros_like(inputs[0]))):
            results = model(value)
            np.savez(references / f"case-{ordinal}.npz", in0=value.numpy(),
                **{f"out{i}": result.numpy() for i, result in enumerate(results)})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, choices=(13, 16, 32), default=16)
    arguments = parser.parse_args()
    print(capture(arguments.out, tokens=arguments.tokens))
