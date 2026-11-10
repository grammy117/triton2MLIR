"""A source-captured mixed FP8 contraction with runtime dimensions and K tiles."""

from pathlib import Path

import numpy as np

from .dynamic_abi import GridExtent
from .effect_bundle import write_effect_bundle
from .examples.fp8_dot_kernels import fp8_tiled_dot
from .examples.fp8_dot_reference import inputs, reference
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
    lhs="f8E4M3FN",
    rhs="f8E5M2",
):
    out = Path(out)
    if lhs not in TRITON_TYPES or rhs not in TRITON_TYPES:
        raise ValueError("FP8 contraction requires two pinned FP8 formats")
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    work.mkdir(parents=True, exist_ok=False)
    source = work / "source.ttir"
    source.write_text(
        capture_ttir(
            fp8_tiled_dot,
            {
                "x": "*" + TRITON_TYPES[lhs],
                "y": "*" + TRITON_TYPES[rhs],
                "initial": "*fp32",
                "out": "*fp32",
                "m": "i32",
                "n": "i32",
                "k": "i32",
                "BM": "constexpr",
                "BN": "constexpr",
                "BK": "constexpr",
                "PRECISION": "constexpr",
            },
            {"BM": 16, "BN": 16, "BK": 32, "PRECISION": "ieee"},
            source_target=("hip", "gfx942", 64),
        )
    )
    abi = TensorABI(
        "fp8_tiled_dot",
        (
            TensorBuffer("x", ("m", "k"), dtype=lhs),
            TensorBuffer("y", ("k", "n"), dtype=rhs),
            TensorBuffer("initial", ("m", "n")),
            TensorBuffer("out", ("m", "n"), "write"),
        ),
        grid=(GridExtent("m", 16), GridExtent("n", 16)),
        runtime_scalars=(
            RuntimeScalar("m", "i32", 4, bounds=(0, 35)),
            RuntimeScalar("n", "i32", 5, bounds=(0, 21)),
            RuntimeScalar("k", "i32", 6, bounds=(0, 65)),
        ),
    )
    result = convert_shared(
        source,
        abi,
        work / "translate",
        optimizer=optimizer,
        memory_profile=memory_profile,
        address_policy=address_policy,
        effect_profile=effect_profile,
        f32_atomic_add=f32_atomic_add,
    )
    m, n, k = 17, 19, 37
    a, b, c = inputs(lhs, rhs, m, n, k)
    policy = {
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
            a,
            b,
            c,
            np.asarray(m, np.int32),
            np.asarray(n, np.int32),
            np.asarray(k, np.int32),
        ],
        policy,
        references=[reference(a, b, c, lhs, rhs)],
        work=work / "native",
        python=python,
    )
    return {
        "bundle": str(out),
        "kernel": "fp8_tiled_dot",
        "input_formats": [lhs, rhs],
        "accumulator": "original f32 C",
        "tiles": [16, 16, 32],
        "runtime_dimensions": [m, n, k],
        "native_execution": "passed",
        "merlin_execution": "requires separate qualify-merlin invocation",
    }
