"""Numeric source contracts survive official composite precision rewriting."""

import os

import numpy as np
import pytest
from test_fp8_casts import MERLIN
from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import StringAttr, TensorType

from t2m.dynamic_abi import GridExtent
from t2m.fp8_casts import POLICIES, SOURCE_POLICY
from t2m.frontend import capture_ttir
from t2m.ir import text
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import (
    TensorABI,
    TensorBuffer,
    convert_shared,
    opcode,
    parse_shared,
)
from t2m.standard_floats import _FP8_BY_NAME
from t2m.upstream import lower_with_shared


def conversion(root, precision="bf16x3", profile="tensor", *, invalid_contract=None):
    from t2m.examples.dot_precision_kernels import precision_tiled_dot

    path = root / "source.ttir"
    path.write_text(
        capture_ttir(
            precision_tiled_dot,
            {
                name: "*fp32" if i < 4 else "i32" if i < 7 else "constexpr"
                for i, name in enumerate(precision_tiled_dot.arg_names)
            },
            {"BM": 16, "BN": 16, "BK": 32, "PRECISION": precision},
        )
    )
    lower_with_shared(
        path,
        root / "generic",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        stage="source",
    )
    module = parse_shared((root / "generic/lowered.mlir").read_text())
    dot = next(op for op in module.walk() if opcode(op) == "tt.dot")
    operands = []
    for index, (value, policy, target) in enumerate(
        zip(
            tuple(dot.operands),
            POLICIES,
            ("f8E5M2", "f8E4M3FN", "f8E4M3FNUZ"),
            strict=True,
        )
    ):
        target_type = TensorType(_FP8_BY_NAME[target], value.type.get_shape())
        cast = builtin.UnregisteredOp.with_name("tt.fp_to_fp").create(
            operands=[value],
            result_types=[target_type],
            attributes={
                "rounding": builtin.IntegerAttr(1, builtin.i32),
                SOURCE_POLICY: StringAttr(policy),
                "prov.family": StringAttr("quantize"),
                "prov.op": StringAttr("residual-cast"),
                "prov.region_id": StringAttr(f"cast{index}"),
            },
        )
        if index == 0 and invalid_contract:
            if invalid_contract == "missing":
                del cast.attributes[SOURCE_POLICY]
            else:
                cast.attributes[SOURCE_POLICY] = (
                    StringAttr("unknown")
                    if invalid_contract == "unknown"
                    else builtin.IntegerAttr(1, builtin.i32)
                )
        wide = arith.ExtFOp(cast.results[0], value.type)
        added = arith.AddfOp(value, wide.result)
        dot.parent.insert_ops_before([cast, wide, added], dot)
        operands.append(added.result)
    dot.operands = operands
    dot.attributes.update(
        {
            "prov.family": StringAttr("contraction"),
            "prov.op": StringAttr("mixed-quantized-dot"),
            "prov.region_id": StringAttr("source-dot"),
        }
    )
    path.write_text(text(module))
    spec = TensorABI(
        "precision_tiled_dot",
        (
            TensorBuffer("x", ("m", "k")),
            TensorBuffer("y", ("k", "n")),
            TensorBuffer("initial", ("m", "n")),
            TensorBuffer("out", ("m", "n"), "write"),
        ),
        grid=(GridExtent("m", 16), GridExtent("n", 16)),
        runtime_scalars=tuple(
            RuntimeScalar(name, "i32", i + 4, bounds=(0, maximum))
            for i, (name, maximum) in enumerate((("m", 35), ("n", 21), ("k", 65)))
        ),
    )
    return convert_shared(
        path,
        spec,
        root / "translation",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
        fp8_overflow="per-cast",
    )


def reference(a, b, c, precision):
    from t2m.examples.dot_precision_reference import reference as dot
    from t2m.examples.fp8_cast_reference import reference as cast
    from t2m.tensor_storage import comparison_values

    def quantize(value, target, policy):
        return comparison_values(cast(value, "f32", target, "rtne", policy), target)

    state = c.copy()
    for start in range(0, a.shape[1], 32):
        x, y = a[:, start : start + 32], b[start : start + 32, :]
        with np.errstate(invalid="ignore", over="ignore"):
            x = x + quantize(x, "f8E5M2", "ieee")
            y = y + quantize(y, "f8E4M3FN", "satfinite")
            state = state + quantize(state, "f8E4M3FNUZ", "satfinite-nan-inf")
        state = dot(x, y, state, precision)
    return state


@pytest.mark.skipif(
    not MERLIN, reason="existing native/shared/Merlin environments required"
)
@pytest.mark.parametrize("invalid", ["missing", "unknown", "wrong-type"])
def test_original_numeric_contracts_refuse_before_composite_normalization(
    tmp_path, monkeypatch, invalid
):
    import t2m.shared_tensor as shared
    from t2m.diagnostics import ConversionError

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "a native normalization ran before original numeric contracts validated"
        )

    monkeypatch.setattr(shared, "expand_tensor_pointers", forbidden)
    with pytest.raises(ConversionError, match="explicit valid numeric policy"):
        conversion(tmp_path, invalid_contract=invalid)


@pytest.mark.skipif(
    not MERLIN, reason="existing native/shared/Merlin environments required"
)
@pytest.mark.parametrize(
    "precision,count", [("bf16x3", 3), ("bf16x6", 6), ("tf32", 1), ("tf32x3", 3)]
)
@pytest.mark.parametrize("profile", ("tensor", "buffer", "guarded-buffer"))
def test_protected_mixed_casts_and_composite_precision_execute(
    tmp_path, precision, count, profile
):
    from test_mixed_effects import qualify

    from t2m.tensor_storage import assert_ieee_words

    result = conversion(tmp_path, precision, profile)
    assert result.trace["schema"] == "t2m.shared-tensor-provenance.v2"
    assert (
        len(
            [
                op
                for op in result.trace["source_operations"]
                if op["opcode"] == "tt.fp_to_fp"
            ]
        )
        == 3
    )
    assert (
        len([op for op in result.module.walk() if op.name == "linalg.matmul"]) == count
    )
    original = {row["id"]: row for row in result.trace["source_operations"]}
    dot = next(row for row in original.values() if row["opcode"] == "tt.dot")
    assert dot["dot_precision"]["input_precision"] == precision
    assert dot["intent"]["prov.op"] == "mixed-quantized-dot"
    blocks = {block["id"] for block in result.trace["original_blocks"]}
    assert all(row["block"] in blocks for row in original.values())
    for op in result.module.walk():
        if op.name == "linalg.matmul":
            assert (
                "mixed-quantized-dot"
                in op.attributes["prov.kernel_original_dot_precisions"].data
            )
    for report in result.trace["transformations"]:
        if report.get("source_graph") == "original":
            assert all(source in original for source in report["source_ids"])
    rows, cols, k = 3, 5, 37
    a = ((np.arange(rows * k).reshape(rows, k) % 17 - 8) * 0.12731).astype(np.float32)
    b = ((np.arange(k * cols).reshape(k, cols) % 13 - 6) * 0.08313).astype(np.float32)
    c = ((np.arange(rows * cols).reshape(rows, cols) % 7 - 3) * 0.31419).astype(
        np.float32
    )
    expected = reference(a, b, c, precision)
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    capture, qualified = qualify(
        result,
        tmp_path,
        [a, b, c, *[np.asarray(n, np.int32) for n in (rows, cols, k)]],
        policy,
        [expected],
    )
    with np.load(qualified / "evaluation/outputs.npz", allow_pickle=False) as actual:
        assert_ieee_words(actual["out0"], expected, "f32")
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    library = _record(qualified / "host/model.so")
    for index, (rows, cols, k) in enumerate(((0, 5, 65), (2, 3, 0), (2, 3, 65))):
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        left = ((np.arange(rows * k).reshape(rows, k) % 17 - 8) * 0.12731).astype(
            np.float32
        )
        right = ((np.arange(k * cols).reshape(k, cols) % 13 - 6) * 0.08313).astype(
            np.float32
        )
        initial = (
            (np.arange(rows * cols).reshape(rows, cols) % 7 - 3) * 0.31419
        ).astype(np.float32)
        np.savez(
            caller / "inputs.npz",
            in0=left,
            in1=right,
            in2=initial,
            in3=np.array(rows, np.int32),
            in4=np.array(cols, np.int32),
            in5=np.array(k, np.int32),
        )
        import json

        (caller / "reference.json").write_text(json.dumps(policy))
        np.savez(
            caller / "reference.npz", out0=reference(left, right, initial, precision)
        )
        invocation = invoke_kernel(
            qualified,
            caller / "inputs.npz",
            caller / "reference.json",
            tmp_path / f"replay-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(tmp_path / f"replay-{index}")
        assert invocation["native_compilation"] == "reused-qualified-library"
        assert _record(qualified / "host/model.so") == library
    from test_spike_model import TOOLS as TARGET_TOOLS

    if TARGET_TOOLS:
        from scripts import qualify_spike_effect as E

        options = {
            "merlin_root": os.environ["T2M_MERLIN_ROOT"],
            "python": os.environ["T2M_MERLIN_PYTHON"],
        }
        report = E.qualify(capture, tmp_path / "target", **options)
        assert report["evaluation"]["result_count"] == 1
        assert E.verify(tmp_path / "target", **options) == report
