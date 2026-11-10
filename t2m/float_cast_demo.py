"""Public RTZ source with explicit typed results and rational references."""

from pathlib import Path

import numpy as np

from .dynamic_abi import GridExtent
from .effect_bundle import write_effect_bundle
from .examples.float_cast_kernels import rounded_cast
from .examples.float_cast_reference import inputs, reference
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
):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    work.mkdir(parents=True, exist_ok=False)
    source = work / "source.ttir"
    source.write_text(
        capture_ttir(
            rounded_cast,
            {
                "x": "*fp32",
                "out": "*fp16",
                "n": "i32",
                "B": "constexpr",
                "ROUND": "constexpr",
            },
            {"B": 64, "ROUND": "rtz"},
        )
    )
    abi = TensorABI(
        "rounded_cast",
        (TensorBuffer("x", ("n",)), TensorBuffer("out", ("n",), "write", "f16")),
        grid=(GridExtent("n", 64),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 65536)),),
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
    values = inputs("f32")
    expected = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    write_effect_bundle(
        result,
        out,
        [values, np.asarray(values.size, np.int32)],
        expected,
        references=[reference(values, "f32", "f16", "rtz")],
        work=work / "native",
        python=python,
    )
    return {
        "bundle": str(out),
        "source_rounding": "rtz",
        "source_dtype": "f32",
        "target_dtype": "f16",
        "native_numeric_check": "passed",
        "result_count": 1,
        "consumer_execution": "not-performed",
    }
