"""Composite dot and per-cast policies retain owning C/output state."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest
from test_descriptor_state import REQUIRED
from xdsl.dialects import builtin

from t2m.allocation_bindings import (
    DescriptorArgument,
    PointerArgument,
    ScalarArgument,
    convert_allocations,
)
from t2m.examples.fp8_allocation_dot_reference import reference
from t2m.fp8_casts import POLICIES, SOURCE_POLICY
from t2m.frontend import capture_ttir, read_ttir_generic
from t2m.ir import text
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import opcode, parse_shared
from t2m.strided_memory_demo import capture as capture_abi

pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing native/shared/Merlin environments required"
)


def convert(root, *, descriptors=False, precision="bf16x3", invalid=False):
    from t2m.examples.fp8_allocation_dot_kernels import fp8_allocation_dot

    root.mkdir(parents=True, exist_ok=True)
    _, original = capture_abi(root / "abi", writable=True)
    abi = replace(
        original,
        entry="fp8_allocation_contraction",
        buffers=(*original.buffers[:2], replace(original.buffers[3], name="state")),
        runtime_scalars=(
            *(
                replace(s, source_arg=s.source_arg - 1)
                for s in original.runtime_scalars
            ),
            RuntimeScalar("c_origin", "i64", 18, bounds=(0, 8191)),
        ),
        intent="Original FP8 residuals and composite dot; shared C/output root; retain tiles, K reuse and logical grid",
    )
    source = root / "source.ttir"
    signature = {
        "x": "tensordesc<fp32[16,32]>" if descriptors else "*fp32",
        "y": "tensordesc<fp32[32,16]>" if descriptors else "*fp32",
        "initial": "*fp32",
        "out": "*fp32",
        **{scalar.name: scalar.dtype for scalar in original.runtime_scalars},
    }
    constants = {
        "BM": 16,
        "BN": 16,
        "BK": 32,
        "PRECISION": precision,
        "DESCRIPTORS": descriptors,
    }
    signature.update({name: "constexpr" for name in constants})
    source.write_text(
        capture_ttir(
            fp8_allocation_dot, signature, constants, source_target=("cuda", 90, 32)
        )
    )
    read_ttir_generic(source, root / "fixture-print")
    module = parse_shared((root / "fixture-print/lowered.mlir").read_text())
    casts = [
        op
        for op in module.walk()
        if opcode(op) == "tt.fp_to_fp" and "f8" in str(op.results[0].type)
    ]
    assert len(casts) == 3
    for op, policy in zip(casts, POLICIES, strict=True):
        op.attributes[SOURCE_POLICY] = builtin.StringAttr(
            "unknown" if invalid else policy
        )
    (dot,) = [op for op in module.walk() if opcode(op) == "tt.dot"]
    dot.attributes.update(
        {
            "prov.family": builtin.StringAttr("contraction"),
            "prov.op": builtin.StringAttr("residual-quantized-tiled-dot"),
            "prov.region_id": builtin.StringAttr("source-dot"),
        }
    )
    source.write_text(text(module))
    if descriptors:
        bindings = (
            DescriptorArgument("x", ("m", "k"), ("a_row", "a_k"), offset="a_base"),
            *(ScalarArgument(name) for name in ("m", "k", "a_row", "a_k")),
            DescriptorArgument("y", ("k", "n"), ("b_k", "b_col"), offset="b_base"),
            *(ScalarArgument(name) for name in ("k", "n", "b_k", "b_col")),
        )
    else:
        bindings = (PointerArgument("x"), PointerArgument("y"))
    bindings += (
        PointerArgument("state", "c_origin"),
        PointerArgument("state"),
        *(ScalarArgument(s.name) for s in original.runtime_scalars),
    )
    return convert_allocations(
        source,
        abi,
        bindings,
        root / "binding",
        source_entry="fp8_allocation_dot",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer",
        address_policy="source-words",
        fp8_overflow="per-cast",
    )


def arguments(
    m, n, k, precision, *, strides=(79, 2, 64, 3), out_strides=(-29, 2), c_origin=5
):
    from t2m.examples.allocation_state_reference import arguments as layout

    values, _ = layout(m, n, k, strides, out_strides, "f32", c_origin=c_origin)
    x, y, state = values[:3]
    if m and k:
        x[np.isfinite(x)] *= np.float32(0.12731)
    if k and n:
        y[np.isfinite(y)] *= np.float32(0.08313)
    expected = state.copy()
    if m and n:
        left = (
            np.ndarray(
                (m, k),
                np.float32,
                buffer=x,
                offset=int(values[10]) * 4,
                strides=(int(values[6]) * 4, int(values[7]) * 4),
            )
            if k
            else np.empty((m, 0), np.float32)
        )
        right = (
            np.ndarray(
                (k, n),
                np.float32,
                buffer=y,
                offset=int(values[11]) * 4,
                strides=(int(values[8]) * 4, int(values[9]) * 4),
            )
            if k
            else np.empty((0, n), np.float32)
        )
        initial = state[c_origin : c_origin + m * n].reshape(m, n).copy()
        computed = reference(left, right, initial, precision)
        view = np.ndarray(
            (m, n),
            np.float32,
            buffer=expected,
            offset=int(values[16]) * 4,
            strides=(int(values[14]) * 4, int(values[15]) * 4),
        )
        view[...] = computed
    return values, expected


@pytest.mark.parametrize("descriptors", [False, True], ids=["pointer", "descriptor"])
@pytest.mark.parametrize(
    "precision,count", [("bf16x3", 3), ("bf16x6", 6), ("tf32x3", 3)]
)
def test_composite_dot_preserves_source_C_and_irregular_allocation_state(
    tmp_path, descriptors, precision, count
):
    from test_mixed_effects import qualify

    from t2m.examples.strided_memory_reference import policy

    result = convert(tmp_path, descriptors=descriptors, precision=precision)
    protection = json.loads(result.auxiliary_artifacts["allocation-protection.json"])
    (source_dot,) = [
        row
        for row in protection["source_graph"]["operations"]
        if row["opcode"] == "tt.dot"
    ]
    assert source_dot["dot_precision"]["input_precision"] == precision
    assert len(protection["fp8_conversions"]) == 3
    contractions = [op for op in result.module.walk() if op.name == "linalg.matmul"]
    assert len(contractions) == count and all(
        op.results[0].type.get_shape() == (16, 16) for op in contractions
    )
    for op in [next(iter(result.module.ops)), *contractions]:
        intent = json.loads(
            op.attributes["prov.allocation_original_dot_precisions"].data
        )
        assert intent["source_sha256"] == protection["artifacts"]["original"]["sha256"]
        assert intent["grants_semantics"] is False
        (original,) = intent["contractions"].values()
        assert original["input_precision"] == precision
        assert original["intent"]["prov.op"] == "residual-quantized-tiled-dot"
    contracts = json.loads(
        next(iter(result.module.ops))
        .attributes["prov.allocation_source_contracts"]
        .data
    )
    assert contracts["fp8_conversions"] == protection["fp8_conversions"]
    values, expected = arguments(3, 5, 37, precision)
    capture, qualified = qualify(
        result, tmp_path, values, policy(words=True), [expected]
    )
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    library = _record(qualified / "host/model.so")
    for index, (m, n, k) in enumerate(
        ((17, 5, 37), (3, 19, 37), (3, 5, 0), (0, 7, 65))
    ):
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        values, expected = arguments(
            m,
            n,
            k,
            precision,
            strides=(-79, -2, -64, 3),
            out_strides=(n, 1),
            c_origin=7 if m else 8191,
        )
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(values)}
        )
        np.savez(caller / "reference.npz", out0=expected)
        (caller / "reference.json").write_text(json.dumps(policy(words=True)))
        invoke_kernel(
            qualified,
            caller / "inputs.npz",
            caller / "reference.json",
            tmp_path / f"replay-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(tmp_path / f"replay-{index}")
        assert _record(qualified / "host/model.so") == library
    if precision == "bf16x3":
        bad = tmp_path / "invalid"
        bad.mkdir()
        values, expected = arguments(3, 5, 37, precision)
        values[18] = np.asarray(8191, np.int64)
        np.savez(
            bad / "inputs.npz", **{f"in{i}": value for i, value in enumerate(values)}
        )
        np.savez(bad / "reference.npz", out0=expected)
        (bad / "reference.json").write_text(json.dumps(policy(words=True)))
        with pytest.raises(ValueError, match="actual effect outcome differs"):
            invoke_kernel(
                qualified,
                bad / "inputs.npz",
                bad / "reference.json",
                bad / "invocation",
                merlin_root=os.environ["T2M_MERLIN_ROOT"],
                python=os.environ["T2M_MERLIN_PYTHON"],
            )
        assert not (bad / "invocation/invocation.json").exists()
        assert _record(qualified / "host/model.so") == library
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_effect as target

        options = {
            "merlin_root": os.environ["T2M_MERLIN_ROOT"],
            "python": os.environ["T2M_MERLIN_PYTHON"],
        }
        receipt = target.qualify(capture, tmp_path / "target", **options)
        assert target.verify(tmp_path / "target", **options) == receipt


@pytest.mark.parametrize("descriptors", [False, True])
def test_invalid_original_cast_refuses_before_allocation_dot_normalization(
    tmp_path, monkeypatch, descriptors
):
    from t2m import frontend
    from t2m.diagnostics import ConversionError

    def forbidden(*_a, **_kw):
        pytest.fail(
            "invalid source numeric contract reached a native normalization pass"
        )

    monkeypatch.setattr(frontend, "expand_tensor_pointers", forbidden)
    with pytest.raises(ConversionError, match="explicit valid numeric policy"):
        convert(tmp_path, descriptors=descriptors, invalid=True)
