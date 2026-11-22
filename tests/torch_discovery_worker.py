"""Exercise the actual Torch adapter in its independently pinned environment."""

import importlib.util
import json
import sys
from pathlib import Path

import torch

repo = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("transformer_capture_test", repo / "scripts/capture_transformer_model.py")
demo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = demo
spec.loader.exec_module(demo)
mode = sys.argv[1]
torch.set_num_threads(1)
model = demo.TwoBlocks().eval()
original = demo.fused._abstract_fn


def wrapper_math(*args):
    return original(*args) + 0.25


def two_launches(*args):
    original(*args)
    return original(*args)


if mode != "positive":
    demo.fused._abstract_fn = {"wrapper-math": wrapper_math, "two-launches": two_launches}[mode]
try:
    record = demo.discovery.discover(model, (torch.empty(13, 16),))
except ValueError as error:
    if mode == "positive":
        raise
    print(json.dumps({"mode": mode, "refused": str(error)}))
else:
    if mode != "positive":
        raise AssertionError("automatic discovery omitted actual wrapper computation")
    from m2m.ir.kernel_registry import KERNEL_REGISTRY

    assert len(record["calls"]) == 1
    identity = record["calls"][0]["qualname"]
    previous = KERNEL_REGISTRY.get(identity)
    try:
        with demo.discovery.register_discovered(record):
            assert identity in KERNEL_REGISTRY
            raise RuntimeError("check context restoration")
    except RuntimeError:
        pass
    assert KERNEL_REGISTRY.get(identity) is previous
    print(json.dumps(record))
