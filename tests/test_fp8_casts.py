"""FP8 overflow policies, exact source rounding and real Merlin consumption."""

import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.examples.fp8_cast_reference import inputs, reference
from t2m.fp8_casts import expand_fp8_casts
from t2m.ir import text
from t2m.shared_tensor import _source_ids, parse_shared
from t2m.tensor_storage import FP8_NUMPY_NAMES, assert_ieee_words
from test_float_casts import probe_source, kernel_source

NATIVE, OPT = os.environ.get("T2M_NATIVE_PYTHON"), os.environ.get("TRITON_SHARED_OPT")
CAST_NATIVE = NATIVE and os.environ.get("T2M_MERLIN_ROOT")
MERLIN = (
    NATIVE
    and OPT
    and all(
        os.environ.get(key)
        for key in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
    )
)


@pytest.mark.skipif(
    not CAST_NATIVE, reason="existing native SDK and Merlin cast support required"
)
@pytest.mark.parametrize("source", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("target", FP8_NUMPY_NAMES)
@pytest.mark.parametrize("mode", ["rtne", "rtz"])
def test_standard_cast_rounding_matches_rational_palette(
    tmp_path, source, target, mode
):
    from t2m.native import execute_native

    values = inputs(source, target)
    module = parse_shared(probe_source(source, target, mode, values.size))
    reports = expand_fp8_casts(module, _source_ids, overflow="ieee")
    module.verify()
    actual, report = execute_native(
        text(module), [values], tmp_path / "native", python=NATIVE
    )
    assert_ieee_words(actual[0], reference(values, source, target, mode), target)
    assert reports[0]["rounding"] == mode


@pytest.mark.skipif(
    not CAST_NATIVE, reason="existing native SDK and Merlin cast support required"
)
@pytest.mark.parametrize("source", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("target", FP8_NUMPY_NAMES)
def test_standard_mlir_truncation_executes_its_original_rtne_contract(
    tmp_path, source, target
):
    from t2m.native import execute_native

    values = inputs(source, target)
    n = values.size
    module = parse_shared(f"""module {{func.func @cast(%x:tensor<{n}x{source}>)->tensor<{n}x{target}> {{
      %empty=tensor.empty():tensor<{n}x{target}>
      %r=linalg.generic {{indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}}
        ins(%x:tensor<{n}x{source}>) outs(%empty:tensor<{n}x{target}>) {{
        ^bb0(%a:{source},%unused:{target}):
          %v=arith.truncf %a:{source} to {target}
          linalg.yield %v:{target}
      }}->tensor<{n}x{target}>
      return %r:tensor<{n}x{target}>
    }}}}""")
    before = text(module)
    assert expand_fp8_casts(module, _source_ids, overflow="satfinite") == []
    assert text(module) == before
    actual, _ = execute_native(
        text(module), [values], tmp_path / "native", python=NATIVE
    )
    assert_ieee_words(actual[0], reference(values, source, target), target)


@pytest.mark.skipif(
    not CAST_NATIVE, reason="existing native SDK and Merlin cast support required"
)
@pytest.mark.parametrize("target", FP8_NUMPY_NAMES)
@pytest.mark.parametrize("mode", ["rtne", "rtz"])
@pytest.mark.parametrize("overflow", ["satfinite", "satfinite-nan-inf"])
def test_explicit_saturation_and_infinity_policies(tmp_path, target, mode, overflow):
    from t2m.native import execute_native

    values = inputs("f32", target)
    module = parse_shared(probe_source("f32", target, mode, values.size))
    expand_fp8_casts(module, _source_ids, overflow=overflow)
    actual, _ = execute_native(
        text(module), [values], tmp_path / "native", python=NATIVE
    )
    assert_ieee_words(
        actual[0], reference(values, "f32", target, mode, overflow), target
    )


@pytest.mark.skipif(
    not CAST_NATIVE, reason="existing native SDK and Merlin cast support required"
)
@pytest.mark.parametrize(
    "source,target",
    [
        (source, target)
        for source in FP8_NUMPY_NAMES
        for target in FP8_NUMPY_NAMES
        if source != target
    ],
)
@pytest.mark.parametrize("mode", ["rtne", "rtz"])
def test_all_equal_width_format_changes_keep_nonfinite_and_zero_rules(
    tmp_path, source, target, mode
):
    from t2m.native import execute_native

    values = np.arange(256, dtype=np.uint8)
    module = parse_shared(probe_source(source, target, mode, values.size))
    expand_fp8_casts(module, _source_ids, overflow="ieee")
    module.verify()
    actual, _ = execute_native(
        text(module), [values], tmp_path / "native", python=NATIVE
    )
    assert_ieee_words(
        actual[0], reference(values, source, target, mode, "ieee"), target
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing-policy",
        "missing-mode",
        "unknown-mode",
        "wrong-mode-type",
        "property",
        "mixed",
    ],
)
def test_unqualified_source_refuses_before_any_edits(change):
    source = probe_source("f32", "f8E5M2", "rtz", 4)
    if change == "missing-mode":
        source = source.replace("rounding=0:i32", "")
    elif change == "unknown-mode":
        source = source.replace("rounding=0:i32", "rounding=2:i32")
    elif change == "wrong-mode-type":
        source = source.replace("rounding=0:i32", "rounding=0:i64")
    elif change == "property":
        source = source.replace("rounding=0:i32", "rounding=0:i32,unknown=true")
    elif change == "mixed":
        source = source.replace(
            "return %r",
            '%bad="tt.fp_to_fp"(%x) {rounding=9:i32}:(tensor<4xf32>)->tensor<4xf8E5M2> return %r',
        )
    module = parse_shared(source)
    before = text(module)
    with pytest.raises(ConversionError):
        expand_fp8_casts(
            module, _source_ids, overflow=None if change == "missing-policy" else "ieee"
        )
    assert text(module) == before


def test_source_intent_survives_serialization_and_policy_is_executable():
    source = probe_source("f64", "f8E4M3FN", "rtz", 4).replace(
        "rounding=0:i32",
        'rounding=0:i32,prov.family="quantize",prov.op="cast",prov.region_id="tile"',
    )
    module = parse_shared(source)
    expand_fp8_casts(module, _source_ids, overflow="satfinite")
    module = parse_shared(text(module))
    module.verify()
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    for key, value in {
        "family": "quantize",
        "op": "cast",
        "region_id": "tile",
        "fp8_overflow": "satfinite",
    }.items():
        assert generic.attributes["prov." + key].data == value
    assert {
        "arith.maximumf",
        "arith.minimumf",
        "arith.truncf",
        "arith.cmpf",
        "arith.subi",
    } <= {op.name for op in module.walk()}


@pytest.mark.parametrize("dtype", FP8_NUMPY_NAMES)
def test_exact_nan_class_comparison_preserves_every_other_word(dtype):
    words = np.arange(256, dtype=np.uint8)
    assert_ieee_words(words, words.copy(), dtype)
    alternate = words.copy()
    if dtype == "f8E4M3FN":
        alternate[127], alternate[255] = 255, 127
    elif dtype == "f8E5M2":
        alternate[[125, 126, 127, 253, 254, 255]] = 126
    assert_ieee_words(words, alternate, dtype)
    for word in (0, 1, 32, 128):
        changed = words.copy()
        changed[word] ^= 1
        with pytest.raises(AssertionError):
            assert_ieee_words(words, changed, dtype)


@pytest.mark.parametrize("target", FP8_NUMPY_NAMES)
def test_palette_reference_agrees_with_independent_standard_codec(target):
    ml_dtypes = pytest.importorskip("ml_dtypes")

    dtype = getattr(ml_dtypes, FP8_NUMPY_NAMES[target])
    values = inputs("f32", target)
    with np.errstate(over="ignore", invalid="ignore"):
        codec = values.astype(dtype).view(np.uint8)
    assert_ieee_words(reference(values, "f32", target), codec, target)


@pytest.mark.skipif(
    not MERLIN, reason="existing shared/native/Merlin environments required"
)
@pytest.mark.parametrize("target", FP8_NUMPY_NAMES)
@pytest.mark.parametrize(
    "mode,overflow,profile",
    [
        ("rtne", "ieee", "tensor"),
        ("rtz", "satfinite", "buffer"),
        ("rtne", "satfinite-nan-inf", "guarded-buffer"),
    ],
)
def test_ttir_fp8_conversion_reaches_working_merlin_interface(
    tmp_path, target, mode, overflow, profile
):
    from test_mixed_effects import qualify
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    path = tmp_path / "source.ttir"
    path.write_text(kernel_source("f32", target, mode))
    abi = TensorABI(
        "cast_kernel",
        (
            TensorBuffer("x", ("n",)),
            TensorBuffer("out", ("n",), "write", target),
            TensorBuffer("raw", ("n",), "write"),
        ),
        grid=(GridExtent("n", 64),),
        runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 65536)),),
    )
    result = convert_shared(
        path,
        abi,
        tmp_path / "translation",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
        fp8_overflow=overflow,
    )
    values = inputs("f32", target)
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {
            "archive": "reference.npz",
            "comparisons": [{"kind": "ieee"}, {"kind": "words"}],
        },
    }
    qualify(
        result,
        tmp_path,
        [values, np.asarray(values.size, np.int32)],
        policy,
        [reference(values, "f32", target, mode, overflow), values.copy()],
    )


@pytest.mark.skipif(
    not MERLIN, reason="existing shared/native/Merlin environments required"
)
@pytest.mark.parametrize("target", FP8_NUMPY_NAMES)
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_real_source_loop_preserves_each_rounding_boundary_in_reused_library(
    tmp_path, target, profile
):
    import json
    from test_mixed_effects import qualify
    from t2m.examples.float_cast_kernels import rounded_cast_steps
    from t2m.examples.fp8_cast_reference import recurrence
    from t2m.dynamic_abi import GridExtent
    from t2m.fp8_demo import TRITON_TYPES
    from t2m.frontend import capture_ttir
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    path = tmp_path / "source.ttir"
    path.write_text(
        capture_ttir(
            rounded_cast_steps,
            {
                "x": "*fp32",
                "out": "*" + TRITON_TYPES[target],
                "n": "i32",
                "steps": "i32",
                "delta": "fp32",
                "B": "constexpr",
                "ROUND": "constexpr",
            },
            {"B": 64, "ROUND": "rtz"},
            source_target=("hip", "gfx942", 64),
        )
    )
    abi = TensorABI(
        "rounded_cast_steps",
        (TensorBuffer("x", ("n",)), TensorBuffer("out", ("n",), "write", target)),
        grid=(GridExtent("n", 64),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 2, bounds=(0, 65536)),
            RuntimeScalar("steps", "i32", 3, bounds=(0, 3)),
            RuntimeScalar("delta", "f32", 4),
        ),
    )
    result = convert_shared(
        path,
        abi,
        tmp_path / "translation",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
        fp8_overflow="satfinite",
    )
    assert "scf.for" in result.mlir_text
    values = inputs("f32", target)
    delta = np.float32(0.015625)
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    _, qualified = qualify(
        result,
        tmp_path,
        [
            values,
            np.asarray(values.size, np.int32),
            np.asarray(2, np.int32),
            np.asarray(delta, np.float32),
        ],
        policy,
        [recurrence(values, delta, 2, target, "rtz", "satfinite")],
    )
    for index, (n, steps, increment) in enumerate(
        [(0, 3, -delta), (1, 0, delta), (65, 1, -delta), (257, 3, delta)]
    ):
        caller = tmp_path / f"caller{index}"
        caller.mkdir()
        # Unit values distinguish repeated rounding from one final cast.
        x = np.resize(
            np.asarray([1.0, -1.0, 1.0625, -1.0625, -0.0, 0.001], np.float32), n
        )
        np.savez(
            caller / "inputs.npz",
            in0=x,
            in1=np.asarray(n, np.int32),
            in2=np.asarray(steps, np.int32),
            in3=np.asarray(increment, np.float32),
        )
        (caller / "reference.json").write_text(json.dumps(policy))
        np.savez(
            caller / "reference.npz",
            out0=recurrence(x, increment, steps, target, "rtz", "satfinite"),
        )
        out = tmp_path / f"invoke{index}"
        report = invoke_kernel(
            qualified,
            caller / "inputs.npz",
            caller / "reference.json",
            out,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
