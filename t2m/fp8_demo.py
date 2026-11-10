"""A real FP8 source kernel with exact byte references and runtime state."""

from pathlib import Path

import numpy as np

from .dynamic_abi import GridExtent
from .effect_bundle import write_effect_bundle
from .examples.fp8_kernels import fp8_choose, fp8_widen
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer, convert_shared

TRITON_TYPES = {
    "f8E4M3FN": "fp8e4nv",
    "f8E4M3FNUZ": "fp8e4b8",
    "f8E5M2": "fp8e5",
    "f8E5M2FNUZ": "fp8e5b16",
}


def demo(
    out,
    *,
    optimizer=None,
    python=None,
    memory_profile="tensor",
    address_policy="structured",
    effect_profile="ordinary",
    f32_atomic_add=None,
    dtype="f8E4M3FN",
    target=None,
):
    if dtype not in TRITON_TYPES:
        raise ValueError("FP8 demo requires one of the four pinned TTIR FP8 formats")
    if target is not None and target not in ("f16", "bf16", "f32", "f64"):
        raise ValueError("FP8 widening requires a standard wider floating target")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    work.mkdir(parents=True, exist_ok=False)
    source = work / "source.ttir"
    kernel = fp8_choose if target is None else fp8_widen
    constants = {
        "B": 64,
        **({"SCALAR": False} if target is None else {"TARGET": target}),
    }
    buffers = (
        TensorBuffer("x", ("n",), dtype=dtype),
        TensorBuffer("y", ("n",), dtype=dtype),
        TensorBuffer("out", ("n",), "write", dtype=target or dtype),
        *(
            (TensorBuffer("raw", ("n",), "write", dtype=dtype),)
            if target is not None
            else ()
        ),
    )
    source.write_text(
        capture_ttir(
            kernel,
            {
                "x": "*" + TRITON_TYPES[dtype],
                "y": "*" + TRITON_TYPES[dtype],
                "out": "*"
                + (
                    TRITON_TYPES[dtype]
                    if target is None
                    else {"f16": "fp16", "bf16": "bf16", "f32": "fp32", "f64": "fp64"}[
                        target
                    ]
                ),
                **({"raw": "*" + TRITON_TYPES[dtype]} if target is not None else {}),
                "n": "i32",
                "rounds": "i32",
                **{name: "constexpr" for name in constants},
            },
            constants,
            source_target=("hip", "gfx942", 64),
        )
    )
    abi = TensorABI(
        kernel.__name__,
        buffers,
        grid=(GridExtent("n", 64),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", len(buffers), bounds=(0, 257)),
            RuntimeScalar("rounds", "i32", len(buffers) + 1, bounds=(0, 3)),
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
    )
    x = np.resize(np.arange(256, dtype=np.uint8), 257)
    y = np.resize(np.arange(255, -1, -1, dtype=np.uint8), 257)
    reference = x.copy()
    for step in range(2):
        selected = (np.arange(257) + step) % 3 == 1
        reference[selected] = y[selected]
    expected = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "words"}]},
    }
    references = [reference]
    if target is not None:
        from .examples.fp8_reference import widened

        references = [widened(reference, dtype, target), reference]
        expected["results"]["comparisons"] = [{"kind": "ieee"}, {"kind": "words"}]
    write_effect_bundle(
        result,
        out,
        [x, y, np.asarray(257, np.int32), np.asarray(2, np.int32)],
        expected,
        references=references,
        work=work / "native",
        python=python,
    )
    return {
        "bundle": str(out),
        "dtype": dtype,
        **({"target_dtype": target} if target is not None else {}),
        "native_word_check": "passed",
        "source_frontend": "hip:gfx942:64; AST-to-TTIR only",
        "result_count": len(references),
        "result_policies": expected["results"]["comparisons"],
        "consumer_execution": "not-performed",
    }
