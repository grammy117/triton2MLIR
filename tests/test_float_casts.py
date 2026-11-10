"""RTZ, format changes and double-rounding cases retain executable semantics."""

import os

import numpy as np
import pytest

from t2m.examples.float_cast_reference import inputs, reference
from t2m.float_casts import expand_float_casts
from t2m.ir import text
from t2m.shared_tensor import _source_ids, parse_shared
from t2m.tensor_storage import DTYPES, assert_ieee_words

NATIVE = os.environ.get("T2M_NATIVE_PYTHON")
OPT = os.environ.get("TRITON_SHARED_OPT")
MERLIN = (
    OPT
    and NATIVE
    and all(
        os.environ.get(key)
        for key in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
    )
)
PAIRS = [
    ("f32", "f16"),
    ("f32", "bf16"),
    ("f64", "f32"),
    ("f64", "f16"),
    ("f64", "bf16"),
    ("f16", "bf16"),
    ("bf16", "f16"),
]


def probe_source(source, target, mode, n):
    return f"""module {{func.func @cast(%x:tensor<{n}x{source}>)->tensor<{n}x{target}> {{
      %r="tt.fp_to_fp"(%x) {{rounding={0 if mode == "rtz" else 1}:i32}} :
        (tensor<{n}x{source}>)->tensor<{n}x{target}>
      return %r:tensor<{n}x{target}>
    }}}}"""


@pytest.mark.skipif(not NATIVE, reason="native SDK required")
@pytest.mark.parametrize("source,target", PAIRS)
@pytest.mark.parametrize("mode", ["rtz", "rtne"])
def test_cast_expansion_matches_independent_rational_reference(
    tmp_path, source, target, mode
):
    from t2m.native import execute_native

    # This unit slice exercises the adapter itself, including ordinary RTNE
    # forms that production deliberately leaves to the upstream converter.
    values = inputs(source)
    module = parse_shared(probe_source(source, target, mode, values.size))
    reports = expand_float_casts(module, _source_ids)
    if not reports:
        # Shared's normal tensor arithmetic conversion puts the standard
        # scalar cast inside this elementwise body before bufferization.
        n = values.size
        module = parse_shared(f"""module {{func.func @cast(%x:tensor<{n}x{source}>)->tensor<{n}x{target}> {{
          %empty=tensor.empty():tensor<{n}x{target}>
          %result=linalg.generic {{indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}}
            ins(%x:tensor<{n}x{source}>) outs(%empty:tensor<{n}x{target}>) {{
            ^bb0(%a:{source},%unused:{target}):
              %r=arith.truncf %a:{source} to {target}
              linalg.yield %r:{target}
          }}->tensor<{n}x{target}>
          return %result:tensor<{n}x{target}>
        }}}}""")
    module.verify()
    expected = reference(values, source, target, mode)
    actual, _ = execute_native(
        text(module), [values], tmp_path / "native", python=NATIVE
    )
    assert actual[0].dtype == DTYPES[target]
    assert_ieee_words(actual[0], expected, target)


def kernel_source(source, target, mode):
    return f"""module {{tt.func @cast_kernel(%x:!tt.ptr<{source}>,%out:!tt.ptr<{target}>,
        %raw:!tt.ptr<{source}>,%n:i32) {{
      %pid=tt.get_program_id x:i32
      %block=arith.constant 64:i32
      %base=arith.muli %pid,%block:i32
      %r=tt.make_range {{start=0:i32,end=64:i32}}:tensor<64xi32>
      %b=tt.splat %base:i32->tensor<64xi32>
      %lane=arith.addi %b,%r:tensor<64xi32>
      %nv=tt.splat %n:i32->tensor<64xi32>
      %mask=arith.cmpi slt,%lane,%nv:tensor<64xi32>
      %xp=tt.splat %x:!tt.ptr<{source}>->tensor<64x!tt.ptr<{source}>>
      %op=tt.splat %out:!tt.ptr<{target}>->tensor<64x!tt.ptr<{target}>>
      %rp=tt.splat %raw:!tt.ptr<{source}>->tensor<64x!tt.ptr<{source}>>
      %xa=tt.addptr %xp,%lane:tensor<64x!tt.ptr<{source}>>,tensor<64xi32>
      %oa=tt.addptr %op,%lane:tensor<64x!tt.ptr<{target}>>,tensor<64xi32>
      %ra=tt.addptr %rp,%lane:tensor<64x!tt.ptr<{source}>>,tensor<64xi32>
      %zero=arith.constant dense<0.0>:tensor<64x{source}>
      %value=tt.load %xa,%mask,%zero:tensor<64x!tt.ptr<{source}>>
      %converted="tt.fp_to_fp"(%value) {{rounding={0 if mode == "rtz" else 1}:i32}}:
        (tensor<64x{source}>)->tensor<64x{target}>
      tt.store %oa,%converted,%mask:tensor<64x!tt.ptr<{target}>>
      tt.store %ra,%value,%mask:tensor<64x!tt.ptr<{source}>>
      tt.return
    }}}}"""


@pytest.mark.skipif(
    not MERLIN, reason="existing shared/native/Merlin environments required"
)
@pytest.mark.parametrize("source,target", PAIRS)
@pytest.mark.parametrize("mode", ["rtz", "rtne"])
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_rounding_and_format_casts_reach_reused_merlin_interface(
    tmp_path, source, target, mode, profile
):
    import json
    from test_mixed_effects import qualify
    from t2m.dynamic_abi import GridExtent
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    path = tmp_path / "source.ttir"
    path.write_text(kernel_source(source, target, mode))
    abi = TensorABI(
        "cast_kernel",
        (
            TensorBuffer("x", ("n",), dtype=source),
            TensorBuffer("out", ("n",), "write", target),
            TensorBuffer("raw", ("n",), "write", source),
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
    )
    values = inputs(source)
    expected = reference(values, source, target, mode)
    policies = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {
            "archive": "reference.npz",
            "comparisons": [{"kind": "ieee"}, {"kind": "words"}],
        },
    }
    capture, qualified = qualify(
        result,
        tmp_path,
        [values, np.asarray(values.size, np.int32)],
        policies,
        [expected, values.copy()],
    )
    for i, n in enumerate((0, 1, 65, values.size - 3)):
        caller = tmp_path / f"caller{i}"
        caller.mkdir()
        np.savez(
            caller / "inputs.npz", in0=values[:n].copy(), in1=np.asarray(n, np.int32)
        )
        (caller / "reference.json").write_text(json.dumps(policies))
        np.savez(
            caller / "reference.npz", out0=expected[:n].copy(), out1=values[:n].copy()
        )
        output = tmp_path / f"invoke{i}"
        report = invoke_kernel(
            qualified,
            caller / "inputs.npz",
            caller / "reference.json",
            output,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(output)
        assert report["native_compilation"] == "reused-qualified-library"


@pytest.mark.parametrize(
    "change",
    ["unknown-rounding", "wrong-width", "missing-rounding", "property", "mixed"],
)
def test_unqualified_cast_forms_refuse_before_editing_any_candidate(change):
    from t2m.diagnostics import ConversionError

    source = probe_source("f32", "f16", "rtz", 4)
    if change == "unknown-rounding":
        source = source.replace("rounding=0:i32", "rounding=2:i32")
    elif change == "wrong-width":
        source = source.replace("rounding=0:i32", "rounding=0:i64")
    elif change == "missing-rounding":
        source = source.replace("rounding=0:i32", "")
    elif change == "property":
        source = source.replace(
            "rounding=0:i32", 'rounding=0:i32,unknown.policy="opaque"'
        )
    else:
        source = source.replace(
            "return %r",
            '%bad="tt.fp_to_fp"(%x) {rounding=2:i32}:(tensor<4xf32>)->tensor<4xf16> return %r',
        )
    module = parse_shared(source)
    before = text(module)
    with pytest.raises(ConversionError):
        expand_float_casts(module, _source_ids)
    assert text(module) == before


def test_cast_intent_and_existing_provenance_survive_serialization():
    source = probe_source("bf16", "f16", "rtz", 4).replace(
        "rounding=0:i32",
        'rounding=0:i32,prov.family="source-family",prov.op="source-cast",prov.region_id="source-region"',
    )
    module = parse_shared(source)
    reports = expand_float_casts(module, _source_ids)
    assert reports[0]["rounding"] == "rtz"
    module = parse_shared(text(module))
    module.verify()
    root = next(op for op in module.walk() if op.name == "linalg.generic")
    assert root.attributes["prov.family"].data == "source-family"
    assert root.attributes["prov.op"].data == "source-cast"
    assert root.attributes["prov.region_id"].data == "source-region"
    assert root.attributes["prov.cast_source"].data == "bf16"
    assert root.attributes["prov.cast_target"].data == "f16"
    assert root.attributes["prov.cast_rounding"].data == "rtz"


@pytest.mark.skipif(not OPT, reason="existing shared optimizer required")
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
@pytest.mark.parametrize("mode", ["rtne", "rtz"])
def test_intake_annotations_survive_upstream_and_final_emission(
    tmp_path, profile, mode
):
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar

    path = tmp_path / "source.ttir"
    attrs = ',prov.family="caller-family",prov.op="caller-cast",prov.region_id="caller-region"'
    path.write_text(
        kernel_source("f32", "f16", mode).replace(
            f"rounding={0 if mode == 'rtz' else 1}:i32",
            f"rounding={0 if mode == 'rtz' else 1}:i32" + attrs,
        )
    )
    abi = TensorABI(
        "cast_kernel",
        (
            TensorBuffer("x", ("n",)),
            TensorBuffer("out", ("n",), "write", "f16"),
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
    )
    annotated = [
        op for op in result.module.walk() if "prov.intake_intent" in op.attributes
    ]
    assert annotated
    for op in annotated:
        assert op.attributes["prov.family"].data == "caller-family"
        assert op.attributes["prov.op"].data == "caller-cast"
        assert op.attributes["prov.region_id"].data == "caller-region"


@pytest.mark.skipif(
    not MERLIN, reason="existing shared/native/Merlin environments required"
)
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_public_rounding_demo_has_real_source_and_independent_reference(
    tmp_path, profile
):
    from t2m.cli import shared_demo
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification

    capture, qualified = tmp_path / "capture", tmp_path / "qualified"
    shared_demo(
        capture,
        kernel="rounded-cast",
        optimizer=OPT,
        python=NATIVE,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    assert "tt.fp_to_fp" in (capture / "source.ttir").read_text()
    qualify_kernel(
        capture,
        qualified,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,
        clang=os.environ["T2M_MERLIN_CLANG"],
    )
    verify_kernel_qualification(qualified)


@pytest.mark.skipif(
    not MERLIN, reason="existing shared/native/Merlin environments required"
)
@pytest.mark.parametrize("target", ["f16", "bf16"])
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_each_source_rounding_boundary_survives_runtime_recurrence(
    tmp_path, target, profile
):
    import json
    from test_mixed_effects import qualify
    from t2m.examples.float_cast_kernels import rounded_cast_steps
    from t2m.examples.float_cast_reference import recurrence
    from t2m.frontend import capture_ttir
    from t2m.dynamic_abi import GridExtent
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = tmp_path / "source.ttir"
    source.write_text(
        capture_ttir(
            rounded_cast_steps,
            {
                "x": "*fp32",
                "out": "*" + ("fp16" if target == "f16" else "bf16"),
                "n": "i32",
                "steps": "i32",
                "delta": "fp32",
                "B": "constexpr",
                "ROUND": "constexpr",
            },
            {"B": 64, "ROUND": "rtz"},
        )
    )
    abi = TensorABI(
        "rounded_cast_steps",
        (TensorBuffer("x", ("n",)), TensorBuffer("out", ("n",), "write", target)),
        grid=(GridExtent("n", 64),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 2, bounds=(0, 65)),
            RuntimeScalar("steps", "i32", 3, bounds=(0, 3)),
            RuntimeScalar("delta", "f32", 4),
        ),
    )
    result = convert_shared(
        source,
        abi,
        tmp_path / "translation",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    assert "scf.for" in result.mlir_text

    def case(n, steps, delta):
        x = np.resize(np.asarray([1.0, -1.0, 0.0, -0.0, 2.0, -2.0], np.float32), n)
        d = np.asarray(delta, np.float32)
        return [x, np.asarray(n, np.int32), np.asarray(steps, np.int32), d], [
            recurrence(x, d, steps, target)
        ]

    delta = 0.0006 if target == "f16" else 0.003
    args, values = case(65, 3, delta)
    expected = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    capture, qualified = qualify(result, tmp_path, args, expected, values)
    for i, (n, steps, d) in enumerate(
        ((0, 0, 0.1), (1, 0, 0.1), (17, 1, -delta), (65, 2, 2 * delta))
    ):
        args, values = case(n, steps, d)
        caller = tmp_path / f"caller{i}"
        caller.mkdir()
        np.savez(
            caller / "inputs.npz", **{f"in{j}": value for j, value in enumerate(args)}
        )
        (caller / "reference.json").write_text(json.dumps(expected))
        np.savez(caller / "reference.npz", out0=values[0])
        output = tmp_path / f"invoke{i}"
        report = invoke_kernel(
            qualified,
            caller / "inputs.npz",
            caller / "reference.json",
            output,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(output)
        assert report["native_compilation"] == "reused-qualified-library"


@pytest.mark.skipif(not NATIVE, reason="native SDK required")
@pytest.mark.parametrize("source", ["f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ"])
def test_fp8_identity_and_exact_rtz_widening_keep_their_distinct_contracts(
    tmp_path, source
):
    from t2m.native import execute_native
    from t2m.examples.fp8_reference import widened

    words = np.arange(256, dtype=np.uint8)
    for target, expansion in (
        (source, "same-format-word-identity"),
        ("f32", "standard-exact-extension"),
    ):
        module = parse_shared(probe_source(source, target, "rtz", 256))
        reports = expand_float_casts(module, _source_ids)
        assert reports[0]["expansion"] == expansion
        module.verify()
        actual, _ = execute_native(
            text(module), [words], tmp_path / target, python=NATIVE
        )
        if target == source:
            np.testing.assert_array_equal(actual[0], words)
        else:
            assert_ieee_words(actual[0], widened(words, source, "f32"), "f32")


def test_unrelated_dynamic_standard_cast_is_not_reclassified():
    module = parse_shared("""module {func.func @f(%x:tensor<?xf32>)->tensor<?xf16> {
      %r=arith.truncf %x:tensor<?xf32> to tensor<?xf16> return %r:tensor<?xf16>
    }}""")
    before = text(module)
    assert expand_float_casts(module, _source_ids) == []
    assert text(module) == before
