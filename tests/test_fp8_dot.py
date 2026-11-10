"""FP8 contractions retain named tiles and source C through a real consumer."""

import json
import os

import numpy as np
import pytest

from t2m.contractions import PRECISIONS, dot_contract
from t2m.diagnostics import ConversionError
from t2m.examples.fp8_dot_reference import inputs, reference
from t2m.examples.fp8_reference import decode
from t2m.fp8_demo import TRITON_TYPES
from t2m.shared_tensor import opcode, parse_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = (
    OPT
    and NATIVE
    and all(
        os.environ.get(name)
        for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
    )
)
CASES = [(lhs, rhs, "tensor") for lhs in TRITON_TYPES for rhs in TRITON_TYPES] + [
    (dtype, dtype, profile)
    for dtype in TRITON_TYPES
    for profile in ("buffer", "guarded-buffer")
]


def contract_source(dtype, precision):
    acc = "i32" if dtype == "i8" else "f32"
    return f"""module {{func.func @dot(%a:tensor<3x4x{dtype}>,%b:tensor<4x5x{dtype}>,%c:tensor<3x5x{acc}>)->tensor<3x5x{acc}> {{
      %r="tt.dot"(%a,%b,%c) {{inputPrecision={precision}:i32,maxNumImpreciseAcc=0:i32}}:
        (tensor<3x4x{dtype}>,tensor<4x5x{dtype}>,tensor<3x5x{acc}>)->tensor<3x5x{acc}>
      return %r:tensor<3x5x{acc}>
    }}}}"""


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f64", "i8", *TRITON_TYPES])
@pytest.mark.parametrize("precision", PRECISIONS)
def test_non_f32_dot_preserves_ignored_input_precision(dtype, precision):
    module = parse_shared(contract_source(dtype, precision))
    op = next(op for op in module.walk() if opcode(op) == "tt.dot")
    record = dot_contract(op, "source-dot")
    assert record["input_precision"] == PRECISIONS[precision]
    assert record["input_precision_applies"] is False
    assert record["portable_accumulator"] == "original-source-C"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_f32_precision_cannot_be_silently_replaced(precision):
    module = parse_shared(contract_source("f32", precision))
    op = next(op for op in module.walk() if opcode(op) == "tt.dot")
    if precision == 2:
        assert dot_contract(op, "source-dot")["input_precision_applies"]
    else:
        with pytest.raises(ConversionError, match="f32 dot precision"):
            dot_contract(op, "source-dot")


@pytest.mark.parametrize("splat", [False, True])
@pytest.mark.parametrize("negative", [False, True])
def test_zero_accumulators_match_dense_and_scalar_splat_without_losing_sign(
    splat, negative
):
    from t2m.contractions import zero_accumulator

    zero = "-0.0" if negative else "0.0"
    payload = (
        f'%z=arith.constant {zero}:f32 %c="tt.splat"(%z):(f32)->tensor<3x5xf32>'
        if splat
        else f"%c=arith.constant dense<{zero}>:tensor<3x5xf32>"
    )
    module = parse_shared(
        f"module {{func.func @f()->tensor<3x5xf32> {{{payload} return %c:tensor<3x5xf32>}}}}"
    )
    function = next(op for op in module.walk() if op.name == "func.func")
    assert zero_accumulator(function.body.block.last_op.operands[0]) == (
        "negative" if negative else "positive"
    )


@pytest.mark.parametrize(
    "dtype,word,expected",
    [
        ("f32", -2147483648, "negative"),
        ("f32", 0, "positive"),
        ("f8E4M3FN", -128, "negative"),
        ("f8E4M3FNUZ", -128, None),
        ("f8E5M2FNUZ", 0, "positive"),
    ],
)
def test_bitcast_zero_proofs_distinguish_fnuz_nan(dtype, word, expected):
    from t2m.contractions import zero_accumulator

    width = 32 if dtype == "f32" else 8
    module = parse_shared(f"""module {{func.func @f()->tensor<3x5x{dtype}> {{
      %bits=arith.constant dense<{word}>:tensor<3x5xi{width}>
      %r="tt.bitcast"(%bits):(tensor<3x5xi{width}>)->tensor<3x5x{dtype}>
      return %r:tensor<3x5x{dtype}>
    }}}}""")
    function = next(op for op in module.walk() if op.name == "func.func")
    assert zero_accumulator(function.body.block.last_op.operands[0]) == expected


def skipped_zero_program(*, invalid=False):
    extra = (
        """%one=arith.constant 1.0:f32
      %badseed=linalg.fill ins(%one:f32) outs(%empty:tensor<2x2xf32>)->tensor<2x2xf32>
      %other=linalg.matmul ins(%a,%b:tensor<2x2xf32>,tensor<2x2xf32>) outs(%badseed:tensor<2x2xf32>)->tensor<2x2xf32> loc("ttir.op.0002")"""
        if invalid
        else ""
    )
    return f"""module {{func.func @f(%a:tensor<2x2xf32>,%b:tensor<2x2xf32>,%bonus:tensor<2x2xf32>)->tensor<2x2xf32> {{
      %zero=arith.constant 0.0:f32
      %empty=tensor.empty():tensor<2x2xf32>
      %seed=linalg.fill ins(%zero:f32) outs(%empty:tensor<2x2xf32>)->tensor<2x2xf32>
      %r=linalg.matmul ins(%a,%b:tensor<2x2xf32>,tensor<2x2xf32>) outs(%seed:tensor<2x2xf32>)->tensor<2x2xf32> loc("ttir.op.0000")
      %post=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}}
        ins(%r,%bonus:tensor<2x2xf32>,tensor<2x2xf32>) outs(%empty:tensor<2x2xf32>) {{
          ^bb0(%x:f32,%y:f32,%unused:f32): %v=arith.addf %x,%y:f32 linalg.yield %v:f32
        }}->tensor<2x2xf32> loc("ttir.op.0001")
      {extra}
      return %post:tensor<2x2xf32>
    }}}}"""


def negative_record(index):
    return {
        "id": f"ttir.op.{index:04d}",
        "restore_matmul_accumulator": True,
        "dot_numeric": {"source_accumulator_zero": "negative"},
    }


def test_all_accumulator_candidates_validate_before_any_edits():
    from t2m.ir import text
    from t2m.shared_tensor import _restore_accumulators

    module = parse_shared(skipped_zero_program(invalid=True))
    before = text(module)
    with pytest.raises(ConversionError, match="positive-zero seed"):
        _restore_accumulators(module, [negative_record(0), negative_record(2)])
    assert text(module) == before


@pytest.mark.skipif(not NATIVE, reason="existing native SDK required")
def test_skipped_dense_negative_zero_is_restored_without_moving_downstream_add(
    tmp_path,
):
    from t2m.ir import text
    from t2m.native import execute_native
    from t2m.shared_tensor import _restore_accumulators

    module = parse_shared(skipped_zero_program())
    reports = _restore_accumulators(module, [negative_record(0)])
    module.verify()
    assert reports[0]["kind"] == "restore-source-negative-zero-accumulator"
    assert any(op.name == "linalg.generic" for op in module.walk())
    for index, (a, b, bonus, expected) in enumerate(
        [(-1.0, 0.0, -0.0, -0.0), (1.0, 1.0, 2.0**24, 2.0**24 + 2.0)]
    ):
        arguments = [np.full((2, 2), value, np.float32) for value in (a, b, bonus)]
        output, _ = execute_native(
            text(module), arguments, tmp_path / f"call{index}", python=NATIVE
        )
        np.testing.assert_array_equal(
            output[0].view(np.uint32),
            np.full((2, 2), expected, np.float32).view(np.uint32),
        )


@pytest.mark.skipif(
    not MERLIN, reason="existing shared/native/Merlin environments required"
)
@pytest.mark.parametrize("dtype", TRITON_TYPES)
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_constant_negative_zero_c_and_downstream_add_remain_distinct(
    tmp_path, dtype, profile
):
    from test_mixed_effects import qualify
    from t2m.examples.fp8_dot_kernels import fp8_negative_zero_dot
    from t2m.frontend import capture_ttir
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    path = tmp_path / "source.ttir"
    path.write_text(
        capture_ttir(
            fp8_negative_zero_dot,
            {
                "x": "*" + TRITON_TYPES[dtype],
                "y": "*" + TRITON_TYPES[dtype],
                "out": "*fp32",
                "bonus": "fp32",
                "M": "constexpr",
                "N": "constexpr",
                "K": "constexpr",
            },
            {"M": 16, "N": 16, "K": 32},
            source_target=("hip", "gfx942", 64),
        )
    )
    abi = TensorABI(
        "fp8_negative_zero_dot",
        (
            TensorBuffer("x", (16, 32), dtype=dtype),
            TensorBuffer("y", (32, 16), dtype=dtype),
            TensorBuffer("out", (16, 16), "write"),
        ),
        runtime_scalars=(RuntimeScalar("bonus", "f32", 3),),
    )
    result = convert_shared(
        path,
        abi,
        tmp_path / "convert",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    assert any(
        item["kind"]
        in (
            "restore-source-negative-zero-accumulator",
            "restore-source-contraction-accumulator",
        )
        for item in result.trace["transformations"]
    )
    contract = next(
        record["dot_numeric"]
        for record in result.trace["source_operations"]
        if "dot_numeric" in record
    )
    assert contract["source_accumulator_zero"] == "negative"
    a = np.full(
        (16, 32),
        next(word for word in range(256) if decode(word, dtype) == -1.0),
        np.uint8,
    )
    b = np.zeros((32, 16), np.uint8)
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
        [a, b, np.asarray(-0.0, np.float32)],
        policy,
        [np.full((16, 16), -0.0, np.float32)],
    )
    # With +1 products the source contraction is 32, followed by this bonus.
    # Moving the bonus into C would instead lose every +1 at the f32 boundary.
    one = next(word for word in range(256) if decode(word, dtype) == 1.0)
    caller = tmp_path / "caller"
    caller.mkdir()
    np.savez(
        caller / "inputs.npz",
        in0=np.full_like(a, one),
        in1=np.full_like(b, one),
        in2=np.asarray(2.0**24, np.float32),
    )
    (caller / "reference.json").write_text(json.dumps(policy))
    np.savez(caller / "reference.npz", out0=np.full((16, 16), 2.0**24 + 32, np.float32))
    output = tmp_path / "invoke"
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


@pytest.mark.skipif(
    not MERLIN, reason="existing shared/native/Merlin environments required"
)
@pytest.mark.parametrize("lhs,rhs,profile", CASES)
def test_mixed_fp8_tiles_reuse_merlin_library_across_runtime_shapes(
    tmp_path, lhs, rhs, profile
):
    from test_mixed_effects import qualify
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.fp8_dot_kernels import fp8_tiled_dot
    from t2m.frontend import capture_ttir
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    path = tmp_path / "source.ttir"
    precision = ("tf32", "bf16x3", "bf16x6", "ieee")[list(TRITON_TYPES).index(lhs)]
    path.write_text(
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
            {"BM": 16, "BN": 16, "BK": 32, "PRECISION": precision},
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
        path,
        abi,
        tmp_path / "convert",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    matmuls = [op for op in result.module.walk() if op.name == "linalg.matmul"]
    assert len(matmuls) == 1 and "scf.for" in result.mlir_text
    contracts = json.loads(matmuls[0].attributes["prov.contraction_contracts"].data)
    assert len(contracts) == 1
    numeric = next(iter(contracts.values()))
    assert numeric["input_formats"] == [lhs, rhs]
    assert (
        numeric["input_precision"] == precision
        and not numeric["input_precision_applies"]
    )
    assert numeric["intake_accumulator_value"]
    assert "restore-source-contraction-accumulator" in [
        item["kind"] for item in result.trace["transformations"]
    ]
    m, n, k = 17, 19, 37
    a, b, initial = inputs(lhs, rhs, m, n, k)
    expected = reference(a, b, initial, lhs, rhs)
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    arrays = [
        a,
        b,
        initial,
        np.asarray(m, np.int32),
        np.asarray(n, np.int32),
        np.asarray(k, np.int32),
    ]
    _, qualified = qualify(result, tmp_path, arrays, policy, [expected])
    for index, (m, n, k) in enumerate(
        [(0, 7, 65), (1, 1, 32), (5, 7, 0), (35, 21, 65)]
    ):
        a, b, initial = inputs(lhs, rhs, m, n, k)
        if index == 1:
            a.fill(next(word for word in range(256) if decode(word, lhs) == 1.0))
            b.fill(next(word for word in range(256) if decode(word, rhs) == 1.0))
            initial.fill(2.0**24)
        caller = tmp_path / f"caller{index}"
        caller.mkdir()
        np.savez(
            caller / "inputs.npz",
            in0=a,
            in1=b,
            in2=initial,
            in3=np.asarray(m, np.int32),
            in4=np.asarray(n, np.int32),
            in5=np.asarray(k, np.int32),
        )
        (caller / "reference.json").write_text(json.dumps(policy))
        np.savez(caller / "reference.npz", out0=reference(a, b, initial, lhs, rhs))
        output = tmp_path / f"invoke{index}"
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


@pytest.mark.skipif(not NATIVE, reason="existing native SDK required")
def test_transactional_repairs_follow_replaced_source_c_dependencies(tmp_path):
    from t2m.ir import text
    from t2m.native import execute_native
    from t2m.shared_tensor import _restore_accumulators

    module = parse_shared("""module {func.func @f(%a:tensor<2x2xf32>,%b:tensor<2x2xf32>,%d:tensor<2x2xf32>,%c:tensor<2x2xf32>)->tensor<2x2xf32> {
      %zero=arith.constant 0.0:f32
      %empty=tensor.empty():tensor<2x2xf32>
      %seed=linalg.fill ins(%zero:f32) outs(%empty:tensor<2x2xf32>)->tensor<2x2xf32>
      %first=linalg.matmul ins(%a,%b:tensor<2x2xf32>,tensor<2x2xf32>) outs(%seed:tensor<2x2xf32>)->tensor<2x2xf32> loc("ttir.op.0000")
      %c1=linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}
        ins(%first,%c:tensor<2x2xf32>,tensor<2x2xf32>) outs(%empty:tensor<2x2xf32>) {
          ^bb0(%x:f32,%y:f32,%unused:f32): %v=arith.addf %x,%y:f32 linalg.yield %v:f32
        }->tensor<2x2xf32> loc("ttir.op.0000")
      %second=linalg.matmul ins(%a,%d:tensor<2x2xf32>,tensor<2x2xf32>) outs(%seed:tensor<2x2xf32>)->tensor<2x2xf32> loc("ttir.op.0001")
      %out=linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}
        ins(%second,%c1:tensor<2x2xf32>,tensor<2x2xf32>) outs(%empty:tensor<2x2xf32>) {
          ^bb0(%x:f32,%y:f32,%unused:f32): %v=arith.addf %x,%y:f32 linalg.yield %v:f32
        }->tensor<2x2xf32> loc("ttir.op.0001")
      return %out:tensor<2x2xf32>
    }}""")
    reports = _restore_accumulators(
        module,
        [
            {"id": f"ttir.op.{index:04d}", "restore_matmul_accumulator": True}
            for index in range(2)
        ],
    )
    assert len(reports) == 2
    module.verify()
    arguments = [np.ones((2, 2), np.float32) for _ in range(3)] + [
        np.full((2, 2), 2.0**24, np.float32)
    ]
    output, _ = execute_native(
        text(module), arguments, tmp_path / "native", python=NATIVE
    )
    np.testing.assert_array_equal(output[0], arguments[3])
