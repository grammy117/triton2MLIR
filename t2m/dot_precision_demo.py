"""A real f32 precision kernel through upstream BF16 composite expansion."""

from pathlib import Path

import numpy as np

from .dynamic_abi import GridExtent
from .effect_bundle import write_effect_bundle
from .examples.dot_precision_kernels import precision_tiled_dot
from .examples.dot_precision_reference import inputs, reference
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer, convert_shared


def demo(
    out,
    *,
    precision="bf16x6",
    optimizer=None,
    python=None,
    memory_profile="tensor",
    address_policy="structured",
    effect_profile="ordinary",
    f32_atomic_add=None,
):
    if precision not in ("bf16x3", "bf16x6"):
        raise ValueError("qualified composite precision must be bf16x3 or bf16x6")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    work.mkdir(parents=True, exist_ok=False)
    source = work / "source.ttir"
    signature = {
        name: "*fp32" if i < 4 else "i32" if i < 7 else "constexpr"
        for i, name in enumerate(precision_tiled_dot.arg_names)
    }
    source.write_text(
        capture_ttir(
            precision_tiled_dot,
            signature,
            {"BM": 16, "BN": 16, "BK": 32, "PRECISION": precision},
        )
    )
    abi = TensorABI(
        "precision_tiled_dot",
        (
            TensorBuffer("x", ("m", "k")),
            TensorBuffer("y", ("k", "n")),
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
    a, b, c = inputs(m, n, k)
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
        [a, b, c, *(np.asarray(value, np.int32) for value in (m, n, k))],
        policy,
        references=[reference(a, b, c, precision)],
        work=work / "native",
        python=python,
    )
    return {
        "bundle": str(out),
        "kernel": abi.entry,
        "input_precision": precision,
        "normalization": "official pre-layout F32DotTC; no target compilation",
        "tiles": [16, 16, 32],
        "runtime_dimensions": [m, n, k],
        "native_execution": "passed",
        "merlin_execution": "requires separate qualify-merlin invocation",
    }
