#!/usr/bin/env python3
"""Capture a whole model binding an original TTIR program and CPU word oracle.

The custom call is implemented by supplied TTIR, not a captured Python JIT
kernel. No GPU execution or Python-float payload preservation is claimed.
"""

import argparse
import sys
from pathlib import Path

import m2m
import torch
from m2m.capture.bundle import write_bundle
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2m.examples.literal_state import literal_words
from t2m.examples.literal_state_reference import update


def capture(out, dtype):
    carrier = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "f64": torch.float64,
    }[dtype]
    bits, sentinel, table = literal_words(dtype)
    signed = {16: torch.int16, 32: torch.int32, 64: torch.int64}[bits]
    qualname = "t2m_demo::literal_state"

    @torch.library.custom_op(qualname, mutates_args=())
    def literal_state(
        x: torch.Tensor, state: torch.Tensor, n: int, steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_words = x.detach().view(signed).numpy().view(f"uint{bits}")
        state_words = state.detach().view(signed).numpy().view(f"uint{bits}")
        assert len(x_words) == len(state_words) == n
        outputs = update(x_words, state_words, table, sentinel, steps)
        return tuple(
            torch.from_numpy(words.view(f"int{bits}")).view(carrier)
            for words in outputs
        )

    @literal_state.register_fake
    def fake(x, state, n, steps):
        return torch.empty_like(state), torch.empty_like(state)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                ((torch.arange(19) % 7 - 3) / 8).to(carrier), requires_grad=False
            )
            self.bias = torch.nn.Parameter(
                ((torch.arange(19) % 5 - 2) / 32).to(carrier), requires_grad=False
            )

        def forward(self, x):
            original = self.weight + self.bias
            first, first_saved = literal_state(x, original, 19, 1)
            second, second_saved = literal_state(x, first, 19, 2)
            empty, _ = literal_state(x[:0], original[:0], 0, 3)
            return second, first, first_saved, second_saved, empty, original

    out = Path(out).resolve()
    previous = KERNEL_REGISTRY.get(qualname)
    KERNEL_REGISTRY[qualname] = KernelSpec(qualname)
    try:
        torch.set_num_threads(1)
        model = Model().eval()
        inputs = (((torch.arange(19) % 11 - 5) / 64).to(carrier),)
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
    parser.add_argument("--dtype", choices=("f16", "bf16", "f32", "f64"), required=True)
    args = parser.parse_args()
    print(capture(args.out, args.dtype))
