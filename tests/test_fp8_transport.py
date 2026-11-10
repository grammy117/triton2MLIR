"""Uncomputed FP8 words survive actual compiler interfaces and reused libraries."""

import os

import numpy as np
import pytest

from t2m.standard_floats import FP8_TYPE_NAMES

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = (
    OPT
    and NATIVE
    and all(
        os.environ.get(name)
        for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
    )
)


def capture(root, dtype, profile, scalar=False):
    from t2m.examples.fp8_kernels import fp8_choose
    from t2m.frontend import capture_ttir
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    path = root / "source.ttir"
    triton_type = {
        "f8E4M3FN": "fp8e4nv",
        "f8E4M3FNUZ": "fp8e4b8",
        "f8E5M2": "fp8e5",
        "f8E5M2FNUZ": "fp8e5b16",
    }[dtype]
    path.write_text(
        capture_ttir(
            fp8_choose,
            {
                "x": "*" + triton_type,
                "y": "*" + triton_type,
                "out": "*" + triton_type,
                "n": "i32",
                "rounds": "i32",
                "B": "constexpr",
                "SCALAR": "constexpr",
            },
            {"B": 64, "SCALAR": scalar},
            source_target=("hip", "gfx942", 64),
        )
    )
    abi = TensorABI(
        "fp8_choose",
        (
            TensorBuffer("x", ("n",), dtype=dtype),
            TensorBuffer("y", ("n",), dtype=dtype),
            TensorBuffer("out", ("n",), "write", dtype=dtype),
        ),
        grid=(GridExtent("n", 1 if scalar else 64),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 3, bounds=(0, 257)),
            RuntimeScalar("rounds", "i32", 4, bounds=(0, 3)),
        ),
    )
    return convert_shared(
        path,
        abi,
        root / "translation",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )


@pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)
@pytest.mark.parametrize("dtype", FP8_TYPE_NAMES)
@pytest.mark.parametrize(
    "profile,scalar",
    [
        ("tensor", False),
        ("buffer", False),
        ("guarded-buffer", False),
        ("guarded-buffer", True),
    ],
)
def test_all_fp8_words_with_runtime_control_and_reused_merlin_library(
    tmp_path, dtype, profile, scalar
):
    from test_integer_products import replay_library

    result = capture(tmp_path, dtype, profile, scalar)
    assert "scf.for" in result.mlir_text
    assert f"tensor<?x{dtype}>" in result.mlir_text
    cases = []
    for n, rounds in ((0, 0), (1, 0), (65, 1), (257, 2), (257, 3), (256, 0)):
        x, y = (
            np.resize(np.arange(256, dtype=np.uint8), n),
            np.resize(np.arange(255, -1, -1, dtype=np.uint8), n),
        )
        expected = x.copy()
        for step in range(rounds):
            selected = (np.arange(n) + step) % 3 == 1
            expected[selected] = y[selected]
        cases.append(
            ([x, y, np.asarray(n, np.int32), np.asarray(rounds, np.int32)], [expected])
        )
    replay_library(tmp_path, result, cases)
    with np.load(tmp_path / "native/outputs.npz", allow_pickle=False) as outputs:
        np.testing.assert_array_equal(outputs["out0"], cases[-1][1][0])
    # Numeric equality deliberately does not claim NaN payload/zero signs.
    # Verify every saved result word independently, including the no-op loop
    # covering all 256 source encodings through the same compiled library.
    for index, (_, [reference]) in enumerate(cases):
        with np.load(
            tmp_path / f"invoke{index}/evaluation/outputs.npz", allow_pickle=False
        ) as outputs:
            np.testing.assert_array_equal(outputs["out0"], reference)


@pytest.mark.parametrize("dtype", FP8_TYPE_NAMES)
def test_fp8_numeric_comparison_decodes_the_declared_format(dtype):
    pytest.importorskip("ml_dtypes")
    from t2m.tensor_storage import comparison_values

    words = np.asarray([0, 0x80, 0x38, 0x7F], np.uint8)
    values = comparison_values(words, dtype)
    one = {"f8E4M3FN": 1.0, "f8E4M3FNUZ": 0.5, "f8E5M2": 0.5, "f8E5M2FNUZ": 0.25}[dtype]
    assert values[2] == one
    assert values[0] == 0
    assert bool(np.isnan(values[1])) == dtype.endswith("FNUZ")
    with pytest.raises(ValueError, match="raw uint8"):
        comparison_values(np.asarray([1.0], np.float32), dtype)


@pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_visible_demo_binds_exact_word_policy_through_merlin(tmp_path, profile):
    import json
    from t2m.fp8_demo import demo
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification

    capture, qualified = tmp_path / "capture", tmp_path / "qualified"
    demo(
        capture,
        optimizer=OPT,
        python=NATIVE,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    qualify_kernel(
        capture,
        qualified,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,
        clang=os.environ["T2M_MERLIN_CLANG"],
    )
    verify_kernel_qualification(qualified)
    assert json.loads((capture / "reference.json").read_text())["results"][
        "comparisons"
    ] == [{"kind": "words"}]
    assert any(
        "raw uint8" in row
        for row in json.loads((capture / "compatibility.json").read_text())[
            "differences"
        ]
    )
    manifest = json.loads((qualified / "compiler/compile-inputs.json").read_text())
    assert "reference.npz" in manifest["excluded_evaluator_members"]


@pytest.mark.parametrize(
    "target", [("hip", "gfx942", 0), ("hip",), ("hip", object(), 64)]
)
def test_capture_rejects_malformed_frontend_descriptor(target):
    from t2m.frontend import capture_ttir
    from t2m.examples.fp8_kernels import fp8_choose

    with pytest.raises(ValueError, match="source target requires"):
        capture_ttir(
            fp8_choose,
            {name: "i32" for name in fp8_choose.arg_names},
            {},
            source_target=target,
        )
