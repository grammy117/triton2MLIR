"""Official pre-layout precision expansion retains algorithm, tiles and C."""

import json
import os

import numpy as np
import pytest
from xdsl.dialects.builtin import StringAttr

from t2m.dynamic_abi import GridExtent
from t2m.diagnostics import ConversionError
from t2m.examples.dot_precision_reference import inputs, reference
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
from t2m.upstream import lower_with_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = (
    OPT
    and NATIVE
    and all(
        os.environ.get(name)
        for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
    )
)
pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


def convert(root, precision, profile, *, negative_bound=False, volatile=False):
    from t2m.examples.dot_precision_kernels import precision_tiled_dot

    source = root / "capture.ttir"
    signature = {
        name: "*fp32" if i < 4 else "i32" if i < 7 else "constexpr"
        for i, name in enumerate(precision_tiled_dot.arg_names)
    }
    source.write_text(
        capture_ttir(
            precision_tiled_dot,
            signature,
            {"BM": 16, "BN": 16, "BK": 32, "PRECISION": precision},
        )
    )
    lower_with_shared(source, root / "generic-source", optimizer=OPT, stage="source")
    module = parse_shared((root / "generic-source/lowered.mlir").read_text())
    dot = next(op for op in module.walk() if opcode(op) == "tt.dot")
    if negative_bound:
        from xdsl.dialects.builtin import IntegerAttr, i32
        dot.properties["maxNumImpreciseAcc"] = IntegerAttr(-1, i32)
    if volatile:
        from xdsl.dialects.builtin import IntegerAttr, i1
        load = next(op for op in module.walk() if opcode(op) == "tt.load")
        load.properties["isVolatile"] = IntegerAttr(1, i1)
    dot.attributes.update(
        {
            "prov.family": StringAttr("contraction"),
            "prov.op": StringAttr("source-algorithm"),
            "prov.region_id": StringAttr("caller-region"),
        }
    )
    source = root / "source.ttir"
    source.write_text(text(module))
    abi = TensorABI(
        "precision_tiled_dot",
        (
            TensorBuffer("x", ("m", "k")),
            TensorBuffer("y", ("k", "n")),
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
    return convert_shared(
        source,
        abi,
        root / "convert",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )


def arguments(m, n, k, *, ones=False):
    a, b, c = inputs(m, n, k)
    if ones:
        a.fill(1.0)
        b.fill(1.0)
        c.fill(2**24)
    return [a, b, c, *(np.asarray(value, np.int32) for value in (m, n, k))]


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
@pytest.mark.parametrize("precision,count", [("bf16x3", 3), ("bf16x6", 6)])
def test_official_composite_reuses_merlin_library_and_retains_source_intent(
    tmp_path, precision, count, profile
):
    from test_mixed_effects import qualify
    from t2m.invocation import invoke_kernel, verify_invocation

    result = convert(tmp_path, precision, profile)
    assert result.trace["schema"] == "t2m.shared-tensor-provenance.v2"
    original = [op for op in result.trace["source_operations"] if "dot_precision" in op]
    assert (
        len(original) == 1
        and original[0]["dot_precision"]["input_precision"] == precision
    )
    assert original[0]["intent"]["prov.op"] == "source-algorithm"
    matmuls = [op for op in result.module.walk() if op.name == "linalg.matmul"]
    assert len(matmuls) == count
    for op in matmuls:
        assert (
            "source-algorithm"
            in op.attributes["prov.kernel_original_dot_precisions"].data
        )
    assert any(
        "tritongpu-F32DotTC{emu-tf32=false}" in item.get("pipeline", [])
        for item in result.trace["transformations"]
    )
    implementation = next(item["dot_precision_implementation"] for item in result.trace["transformations"] if "dot_precision_implementation" in item)
    assert len(implementation["binary_sha256"]) == 64 and implementation["binary_bytes"] > 0
    assert not implementation["source_reference_is_build_receipt"]
    assert "scf.for" in result.mlir_text and "bf16" in result.mlir_text
    policy = {
        "schema": "t2m.effect-reference.v2",
        "termination": "return",
        "stdout": "",
        "stderr": "",
        "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
    }
    first = arguments(17, 19, 37)
    _, qualified = qualify(
        result, tmp_path, first, policy, [reference(*first[:3], precision)]
    )
    for index, (m, n, k, ones) in enumerate(
        [(0, 7, 65, False), (5, 7, 0, False), (1, 1, 32, True), (3, 5, 65, False)]
    ):
        values = arguments(m, n, k, ones=ones)
        expected = reference(*values[:3], precision)
        if ones:
            np.testing.assert_array_equal(
                expected, np.full((m, n), 2**24 + 32, np.float32)
            )
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(values)}
        )
        (caller / "reference.json").write_text(json.dumps(policy))
        np.savez(caller / "reference.npz", out0=expected)
        out = tmp_path / f"invoke-{index}"
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


@pytest.mark.parametrize("precision", ["bf16x3", "bf16x6"])
def test_negative_original_accumulation_bound_cannot_disappear_in_expansion(tmp_path, precision):
    with pytest.raises(ConversionError, match="nonnegative i32"):
        convert(tmp_path, precision, "tensor", negative_bound=True)


def test_original_effect_gate_runs_before_greedy_precision_normalization(tmp_path):
    with pytest.raises(ConversionError, match="volatile observations"):
        convert(tmp_path, "bf16x3", "tensor", volatile=True)
    assert not (tmp_path / "convert/precision-intake").exists()
    from t2m.frontend import expand_tensor_pointers
    with pytest.raises(ConversionError, match="rewrite failed"):
        expand_tensor_pointers(tmp_path / "source.ttir", tmp_path / "native-refusal")
    assert "volatile observations" in (tmp_path / "native-refusal/stderr.log").read_text()
