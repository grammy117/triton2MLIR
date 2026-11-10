"""Public source FP8 quantization with an explicit software numeric contract."""

from pathlib import Path

import numpy as np

from .dynamic_abi import GridExtent
from .effect_bundle import write_effect_bundle
from .examples.float_cast_kernels import rounded_cast_steps
from .examples.fp8_cast_reference import inputs, recurrence
from .fp8_demo import TRITON_TYPES
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer, convert_shared


def demo(
    out,
    *,
    optimizer=None,
    python=None,
    memory_profile="tensor",
    address_policy="structured",
    effect_profile="ordinary",
    f32_atomic_add=None,
    fp8_overflow="satfinite",
    dtype="f8E4M3FN",
    mode="rtz",
):
    out = Path(out)
    if dtype not in TRITON_TYPES or mode not in ("rtz", "rtne"):
        raise ValueError("FP8 demo requires a pinned format and RTNE/RTZ")
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    work.mkdir(parents=True, exist_ok=False)
    source = work / "source.ttir"
    source.write_text(
        capture_ttir(
            rounded_cast_steps,
            {
                "x": "*fp32",
                "out": "*" + TRITON_TYPES[dtype],
                "n": "i32",
                "steps": "i32",
                "delta": "fp32",
                "B": "constexpr",
                "ROUND": "constexpr",
            },
            {"B": 64, "ROUND": mode},
            source_target=("hip", "gfx942", 64),
        )
    )
    abi = TensorABI(
        "rounded_cast_steps",
        (TensorBuffer("x", ("n",)), TensorBuffer("out", ("n",), "write", dtype)),
        grid=(GridExtent("n", 64),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 2, bounds=(0, 65536)),
            RuntimeScalar("steps", "i32", 3, bounds=(0, 3)),
            RuntimeScalar("delta", "f32", 4),
        ),
    )
    result = convert_shared(
        source,
        abi,
        work / "translation",
        optimizer=optimizer,
        memory_profile=memory_profile,
        address_policy=address_policy,
        effect_profile=effect_profile,
        f32_atomic_add=f32_atomic_add,
        fp8_overflow=fp8_overflow,
    )
    values = inputs("f32", dtype)
    steps, delta = 2, np.float32(0.015625)
    reference = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    write_effect_bundle(
        result,
        out,
        [
            values,
            np.asarray(values.size, np.int32),
            np.asarray(steps, np.int32),
            np.asarray(delta, np.float32),
        ],
        reference,
        references=[recurrence(values, delta, steps, dtype, mode, fp8_overflow)],
        work=work / "native",
        python=python,
    )
    return {
        "bundle": str(out),
        "kernel": "rounded_cast_steps",
        "target": dtype,
        "rounding": mode,
        "fp8_overflow": fp8_overflow,
        "source_loop": "per-step rounded FP8 state",
        "native_execution": "passed",
        "merlin_execution": "requires separate qualify-merlin invocation",
    }
