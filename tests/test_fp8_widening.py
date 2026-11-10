"""Actual FP8 casts retain portable intent and qualify every source encoding."""

import json
import os

import numpy as np
import pytest

from t2m.examples.fp8_reference import FORMATS, widened

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = (
    OPT
    and NATIVE
    and all(
        os.environ.get(key)
        for key in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
    )
)


def convert(root, source, target, profile):
    from t2m.examples.fp8_kernels import fp8_widen
    from t2m.fp8_demo import TRITON_TYPES
    from t2m.frontend import capture_ttir
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    path = root / "source.ttir"
    ptr = "*" + TRITON_TYPES[source]
    out = {"f16": "fp16", "bf16": "bf16", "f32": "fp32", "f64": "fp64"}[target]
    path.write_text(
        capture_ttir(
            fp8_widen,
            {
                "x": ptr,
                "y": ptr,
                "out": "*" + out,
                "raw": ptr,
                "n": "i32",
                "rounds": "i32",
                "B": "constexpr",
                "TARGET": "constexpr",
            },
            {"B": 64, "TARGET": target},
            source_target=("hip", "gfx942", 64),
        )
    )
    abi = TensorABI(
        "fp8_widen",
        (
            TensorBuffer("x", ("n",), dtype=source),
            TensorBuffer("y", ("n",), dtype=source),
            TensorBuffer("out", ("n",), "write", target),
            TensorBuffer("raw", ("n",), "write", source),
        ),
        grid=(GridExtent("n", 64),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 4, bounds=(0, 257)),
            RuntimeScalar("rounds", "i32", 5, bounds=(0, 3)),
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


def case(source, target, n, rounds):
    x = np.resize(np.arange(256, dtype=np.uint8), n)
    y = np.resize(np.arange(255, -1, -1, dtype=np.uint8), n)
    raw = x.copy()
    for step in range(rounds):
        selected = (np.arange(n) + step) % 3 == 1
        raw[selected] = y[selected]
    inputs = [x, y, np.asarray(n, np.int32), np.asarray(rounds, np.int32)]
    reference = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {
            "archive": "reference.npz",
            "comparisons": [{"kind": "ieee"}, {"kind": "words"}],
        },
    }
    return inputs, reference, [widened(raw, source, target), raw]


@pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)
@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_complete_fp8_domain_widens_through_native_and_reused_merlin_library(
    tmp_path, source, target, profile
):
    from test_mixed_effects import qualify
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import verify_kernel_qualification

    result = convert(tmp_path, source, target, profile)
    assert "scf.for" in result.mlir_text and "arith.extf" in result.mlir_text
    assert any(
        record["opcode"] == "tt.fp_to_fp"
        for record in result.trace["source_operations"]
    )
    assert f"tensor<?x{source}>" in result.mlir_text
    arguments, reference, results = case(source, target, 256, 0)
    capture, qualified = qualify(result, tmp_path, arguments, reference, results)
    report = json.loads((qualified / "host/compilation.json").read_text())
    assert "lower_fp8_widening" in report["features"]
    library = _record(qualified / "host/model.so")
    for index, (n, rounds) in enumerate(((0, 3), (1, 0), (65, 1), (257, 3))):
        arguments, reference, results = case(source, target, n, rounds)
        caller = tmp_path / f"caller{index}"
        caller.mkdir()
        np.savez(
            caller / "inputs.npz",
            **{f"in{i}": value for i, value in enumerate(arguments)},
        )
        (caller / "reference.json").write_text(json.dumps(reference))
        np.savez(caller / "reference.npz", out0=results[0], out1=results[1])
        output = tmp_path / f"invoke{index}"
        record = invoke_kernel(
            qualified,
            caller / "inputs.npz",
            caller / "reference.json",
            output,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(output)
        assert record["native_compilation"] == "reused-qualified-library"
        assert _record(output / "source/host/model.so") == library
    verify_kernel_qualification(qualified)


@pytest.mark.parametrize("source", FORMATS)
@pytest.mark.parametrize("target", ["f16", "bf16", "f32", "f64"])
def test_rational_value_oracle_agrees_with_independent_standard_codec(source, target):
    ml = pytest.importorskip("ml_dtypes")
    from t2m.tensor_storage import FP8_NUMPY_NAMES, assert_ieee_words

    raw = np.arange(256, dtype=np.uint8)
    expected = widened(raw, source, target)
    decoded = raw.view(getattr(ml, FP8_NUMPY_NAMES[source]))
    if target == "bf16":
        other = decoded.astype(ml.bfloat16).view(np.uint16)
    else:
        other = decoded.astype(
            {"f16": np.float16, "f32": np.float32, "f64": np.float64}[target]
        )
    assert_ieee_words(expected, other, target)


def test_feature_selection_uses_actual_types_and_casts():
    from t2m.fp8_math import requires_widening
    from t2m.shared_tensor import parse_shared

    for source in FORMATS:
        module = parse_shared(
            f"module {{func.func @f(%x:{source})->f32 {{"
            f"%r=arith.extf %x:{source} to f32 return %r:f32}}}}"
        )
        assert requires_widening(module.walk())
        module = parse_shared(
            f"module {{func.func @f(%x:tensor<4x{source}>)->tensor<4x{source}> {{"
            f"return %x:tensor<4x{source}>}}}}"
        )
        assert not requires_widening(module.walk())


@pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)
def test_visible_widening_demo_retains_complete_reference_and_portable_cast(tmp_path):
    from t2m.cli import shared_demo
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification

    capture, qualified = tmp_path / "capture", tmp_path / "qualified"
    report = shared_demo(capture, kernel="fp8-widen", optimizer=OPT, python=NATIVE)
    assert report["result_count"] == 2 and report["target_dtype"] == "f32"
    assert "arith.extf" in (capture / "model.mlir").read_text()
    assert json.loads((capture / "reference.json").read_text())["results"][
        "comparisons"
    ] == [{"kind": "ieee"}, {"kind": "words"}]
    qualify_kernel(
        capture,
        qualified,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,
        clang=os.environ["T2M_MERLIN_CLANG"],
    )
    verify_kernel_qualification(qualified)
