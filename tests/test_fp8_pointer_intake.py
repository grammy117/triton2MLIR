"""Original pointer/descriptor types and cast policies cross native intake."""

import os

import numpy as np
import pytest
from test_fp8_casts import MERLIN
from xdsl.dialects import arith, builtin

from t2m.diagnostics import ConversionError
from t2m.examples.fp8_cast_reference import inputs
from t2m.examples.fp8_cast_reference import reference as cast_reference
from t2m.fp8_casts import POLICIES, SOURCE_POLICY
from t2m.frontend import capture_ttir, read_ttir_generic
from t2m.ir import text
from t2m.shared_tensor import (
    TensorABI,
    TensorBuffer,
    convert_shared,
    opcode,
    parse_shared,
)
from t2m.standard_floats import _FP8_BY_NAME
from t2m.tensor_storage import comparison_values


def convert(root, descriptors=False, profile="tensor", invalid=None):
    from t2m.examples.tensor_kernels import blockptr_add, descriptor_add

    kernel = descriptor_add if descriptors else blockptr_add
    path = root / "source.ttir"
    path.write_text(
        capture_ttir(
            kernel,
            {"x": "*fp32", "out": "*fp32", "N": "constexpr", "B": "constexpr"},
            {"N": 53, "B": 16},
        )
    )
    read_ttir_generic(path, root / "fixture-print")
    module = parse_shared((root / "fixture-print/lowered.mlir").read_text())
    (loaded,) = [
        op for op in module.walk() if opcode(op) in ("tt.load", "tt.descriptor_load")
    ]
    (value,) = loaded.results
    users = tuple(value.uses)
    operations, state = [], value
    targets = ("f8E5M2", "f8E4M3FN", "f8E4M3FNUZ")
    for index, (policy, target) in enumerate(zip(POLICIES, targets, strict=True)):
        cast = builtin.UnregisteredOp.with_name("tt.fp_to_fp").create(
            operands=[value],
            result_types=[
                builtin.TensorType(_FP8_BY_NAME[target], value.type.get_shape())
            ],
            attributes={
                "rounding": builtin.IntegerAttr(int(index != 1), builtin.i32),
                SOURCE_POLICY: builtin.StringAttr(policy),
                "prov.op": builtin.StringAttr("source-policy"),
            },
        )
        if index == 0 and invalid:
            if invalid == "missing":
                del cast.attributes[SOURCE_POLICY]
            else:
                cast.attributes[SOURCE_POLICY] = (
                    builtin.StringAttr("unknown")
                    if invalid == "unknown"
                    else builtin.IntegerAttr(1, builtin.i32)
                )
        wide = arith.ExtFOp(cast.results[0], value.type)
        added = arith.AddfOp(state, wide.result)
        operations.extend((cast, wide, added))
        state = added.result
    loaded.parent.insert_ops_after(operations, loaded)
    for use in users:
        use.operation.operands[use.index] = state
    path.write_text(text(module))
    spec = TensorABI(
        kernel.__name__,
        (TensorBuffer("x", (53,)), TensorBuffer("out", (53,), "write")),
        grid=(6,),
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


@pytest.mark.skipif(
    not MERLIN, reason="existing native/shared/Merlin environments required"
)
@pytest.mark.parametrize(
    "descriptors", [False, True], ids=["block-pointer", "descriptor"]
)
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_original_types_and_mixed_cast_contracts_execute(
    tmp_path, descriptors, profile
):
    from test_mixed_effects import qualify

    result = convert(tmp_path, descriptors, profile)
    assert result.trace["schema"] == "t2m.shared-tensor-provenance.v2"
    original = result.trace["source_operations"]
    assert sum(row["opcode"] == "tt.fp_to_fp" for row in original) == 3
    assert (
        sum(
            row["opcode"]
            == ("tt.make_tensor_descriptor" if descriptors else "tt.make_tensor_ptr")
            for row in original
        )
        == 2
    )
    (printer,) = [
        row
        for row in result.trace["transformations"]
        if row.get("kind") == "native-verified-source-serialization"
    ]
    assert (
        printer["normalization"] == "not-performed"
        and printer["native_generic_roundtrip_verified"] is True
    )
    expected_pass = (
        "triton-rewrite-tensor-descriptor-to-pointer"
        if descriptors
        else "triton-rewrite-tensor-pointer"
    )
    assert any(
        expected_pass in row.get("pipeline", [])
        for row in result.trace["transformations"]
    )
    x = np.resize(inputs("f32", "f8E5M2"), 53)
    expected = x.copy()
    with np.errstate(invalid="ignore", over="ignore"):
        for policy, target, mode in zip(
            POLICIES,
            ("f8E5M2", "f8E4M3FN", "f8E4M3FNUZ"),
            ("rtne", "rtz", "rtne"),
            strict=True,
        ):
            expected += comparison_values(
                cast_reference(x, "f32", target, mode, policy), target
            )
        expected += np.float32(1)
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    capture, _ = qualify(result, tmp_path, [x], policy, [expected])
    assert (
        "source-only generic serialization"
        in (capture / "compatibility.json").read_text()
    )
    from test_spike_model import TOOLS as TARGET_TOOLS

    if TARGET_TOOLS:
        from scripts import qualify_spike_effect as E

        options = {
            "merlin_root": os.environ["T2M_MERLIN_ROOT"],
            "python": os.environ["T2M_MERLIN_PYTHON"],
        }
        report = E.qualify(capture, tmp_path / "target", **options)
        assert E.verify(tmp_path / "target", **options) == report


@pytest.mark.skipif(
    not MERLIN, reason="existing native/shared/Merlin environments required"
)
@pytest.mark.parametrize("invalid", ["missing", "unknown", "wrong-type"])
def test_original_pointer_numeric_contracts_refuse_before_normalization(
    tmp_path, monkeypatch, invalid
):
    import t2m.shared_tensor as shared

    monkeypatch.setattr(
        shared,
        "expand_tensor_pointers",
        lambda *_a, **_kw: pytest.fail("normalization preceded validation"),
    )
    with pytest.raises(ConversionError, match="explicit valid numeric policy"):
        convert(tmp_path, invalid=invalid)


@pytest.mark.skipif(
    not MERLIN, reason="existing native/shared/Merlin environments required"
)
@pytest.mark.parametrize(
    "descriptors,precision,mode,profile",
    [
        (False, "bf16x3", 3, "guarded-buffer"),
        (True, "tf32x3", 1, "tensor"),
    ],
)
def test_original_pointer_tiles_retain_composite_precision_and_k_reuse(
    tmp_path, descriptors, precision, mode, profile
):
    from test_fp8_precision_intake import reference
    from test_mixed_effects import qualify

    from t2m.examples.tensor_kernels import blockptr_matmul, descriptor_matmul

    kernel = descriptor_matmul if descriptors else blockptr_matmul
    m, n, k = (35, 20, 36) if descriptors else (35, 21, 37)
    path = tmp_path / "source.ttir"
    constants = dict(M=m, N=n, K=k, BM=16, BN=16, BK=16)
    path.write_text(
        capture_ttir(
            kernel,
            {
                "x": "*fp32",
                "y": "*fp32",
                "out": "*fp32",
                **{name: "constexpr" for name in constants},
            },
            constants,
        )
    )
    read_ttir_generic(path, tmp_path / "fixture-print")
    module = parse_shared((tmp_path / "fixture-print/lowered.mlir").read_text())
    (dot,) = [op for op in module.walk() if opcode(op) == "tt.dot"]
    operands = []
    for value, policy, target in zip(
        tuple(dot.operands), POLICIES, ("f8E5M2", "f8E4M3FN", "f8E4M3FNUZ"), strict=True
    ):
        cast = builtin.UnregisteredOp.with_name("tt.fp_to_fp").create(
            operands=[value],
            result_types=[
                builtin.TensorType(_FP8_BY_NAME[target], value.type.get_shape())
            ],
            attributes={
                "rounding": builtin.IntegerAttr(1, builtin.i32),
                SOURCE_POLICY: builtin.StringAttr(policy),
            },
        )
        wide = arith.ExtFOp(cast.results[0], value.type)
        residual = arith.AddfOp(value, wide.result)
        dot.parent.insert_ops_before([cast, wide, residual], dot)
        operands.append(residual.result)
    dot.operands = operands
    dot.attributes.pop("inputPrecision", None)
    dot.properties["inputPrecision"] = builtin.IntegerAttr(mode, builtin.i32)
    dot.attributes["prov.op"] = builtin.StringAttr("original-pointer-quantized-dot")
    path.write_text(text(module))
    abi = TensorABI(
        kernel.__name__,
        (
            TensorBuffer("x", (m, k)),
            TensorBuffer("y", (k, n)),
            TensorBuffer("out", (m, n), "write"),
        ),
        grid=(3, 2),
    )
    result = convert_shared(
        path,
        abi,
        tmp_path / "translation",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
        fp8_overflow="per-cast",
    )
    (original_dot,) = [
        row for row in result.trace["source_operations"] if row["opcode"] == "tt.dot"
    ]
    assert original_dot["dot_precision"]["input_precision"] == precision
    assert original_dot["intent"]["prov.op"] == "original-pointer-quantized-dot"
    assert sum(op.name == "linalg.matmul" for op in result.module.walk()) == 3
    assert any(row["opcode"] == "scf.for" for row in result.trace["source_operations"])
    a = ((np.arange(m * k).reshape(m, k) % 17 - 8) * 0.12731).astype(np.float32)
    b = ((np.arange(k * n).reshape(k, n) % 13 - 6) * 0.08313).astype(np.float32)
    expected = np.zeros((m, n), np.float32)
    for start in range(0, k, 16):
        expected = reference(
            a[:, start : start + 16], b[start : start + 16, :], expected, precision
        )
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    capture, _ = qualify(result, tmp_path, [a, b], policy, [expected])
    from test_spike_model import TOOLS as TARGET_TOOLS

    if TARGET_TOOLS:
        from scripts import qualify_spike_effect as E

        options = {
            "merlin_root": os.environ["T2M_MERLIN_ROOT"],
            "python": os.environ["T2M_MERLIN_PYTHON"],
        }
        report = E.qualify(capture, tmp_path / "target", **options)
        assert E.verify(tmp_path / "target", **options) == report
