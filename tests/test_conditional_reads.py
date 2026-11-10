"""Uniform read gates execute exactly while fresh stores retain coverage."""

import os

import numpy as np
import pytest

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = (
    OPT
    and NATIVE
    and all(
        os.environ.get(key)
        for key in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
    )
)
pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_uniform_gate_retains_inactive_fill_and_live_reads(tmp_path, profile):
    from test_integer_products import replay_library
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.conditional_read_kernels import gated_read
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = tmp_path / "source.ttir"
    source.write_text(
        capture_ttir(
            gated_read,
            {
                "x": "*fp32",
                "out": "*fp32",
                "m": "i32",
                "n": "i32",
                "allow": "i1",
                "B": "constexpr",
            },
            {"B": 16},
        )
    )
    abi = TensorABI(
        "gated_read",
        (TensorBuffer("x", ("m", "n")), TensorBuffer("out", ("m", "n"), "write")),
        grid=(GridExtent("m", 16), GridExtent("n", 16)),
        runtime_scalars=(
            RuntimeScalar("m", "i32", 2, bounds=(0, 35)),
            RuntimeScalar("n", "i32", 3, bounds=(0, 21)),
            RuntimeScalar("allow", "i1", 4),
        ),
    )
    result = convert_shared(
        source,
        abi,
        tmp_path / "convert",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    domains = []
    for m, n, allow in [
        (17, 19, True),
        (17, 19, False),
        (35, 21, False),
        (35, 21, True),
        (0, 7, False),
        (7, 0, True),
        (1, 1, False),
    ]:
        x = np.arange(m * n, dtype=np.float32).reshape(m, n) - 11
        expected = x.copy() if allow else np.full((m, n), 7.0, np.float32)
        domains.append(
            (
                [
                    x,
                    np.asarray(m, np.int32),
                    np.asarray(n, np.int32),
                    np.asarray(allow),
                ],
                [expected],
            )
        )
    replay_library(tmp_path, result, domains)
