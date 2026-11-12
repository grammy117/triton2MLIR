"""Original descriptor parameters retain numeric contracts before ABI expansion."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest
from test_fp8_casts import MERLIN
from xdsl.dialects import arith, builtin

from t2m.bundle import _record
from t2m.diagnostics import ConversionError
from t2m.examples.external_descriptor_reference import arguments
from t2m.examples.fp8_cast_reference import reference as cast_reference
from t2m.external_descriptor_demo import convert as descriptor_demo
from t2m.fp8_casts import POLICIES, SOURCE_POLICY
from t2m.frontend import read_ttir_generic
from t2m.invocation import invoke_kernel, verify_invocation
from t2m.ir import text
from t2m.shared_tensor import convert_shared, opcode, parse_shared
from t2m.standard_floats import _FP8_BY_NAME
from t2m.tensor_storage import comparison_values

TARGETS = ("f8E5M2", "f8E4M3FN", "f8E4M3FNUZ")
MODES = ("rtne", "rtz", "rtne")


def convert(root, *, shape=(16,), malformed_abi=None, invalid_policy=False):
    baseline = descriptor_demo(
        root / "baseline",
        kind="load",
        dtype="f32",
        shape=shape,
        optimizer=os.environ["TRITON_SHARED_OPT"],
    )
    source = root / "source.ttir"
    read_ttir_generic(root / "baseline/source.ttir", root / "fixture-print")
    module = parse_shared((root / "fixture-print/lowered.mlir").read_text())
    (loaded,) = [op for op in module.walk() if opcode(op) == "tt.descriptor_load"]
    (value,) = loaded.results
    users = tuple(value.uses)
    state, operations = value, []
    for index, (target, policy, mode) in enumerate(
        zip(TARGETS, POLICIES, MODES, strict=True)
    ):
        cast = builtin.UnregisteredOp.with_name("tt.fp_to_fp").create(
            operands=[value],
            result_types=[
                builtin.TensorType(_FP8_BY_NAME[target], value.type.get_shape())
            ],
            attributes={
                "rounding": builtin.IntegerAttr(int(mode == "rtne"), builtin.i32),
                SOURCE_POLICY: builtin.StringAttr(
                    "unknown" if invalid_policy and index == 0 else policy
                ),
            },
        )
        wide = arith.ExtFOp(cast.results[0], value.type)
        added = arith.AddfOp(state, wide.result)
        operations.extend((cast, wide, added))
        state = added.result
    loaded.parent.insert_ops_after(operations, loaded)
    for use in users:
        use.operation.operands[use.index] = state
    source.write_text(text(module))
    abi = baseline.spec
    if malformed_abi:
        scalars = list(abi.runtime_scalars)
        first = scalars[0]
        assert first.name == "shape0"
        scalars[0] = (
            replace(first, dtype="i32")
            if malformed_abi == "width"
            else replace(first, bounds=(-1, 32))
        )
        abi = replace(abi, runtime_scalars=tuple(scalars))
    return convert_shared(
        source,
        abi,
        root / "translation",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer",
        address_policy="source-words",
        fp8_overflow="per-cast",
    )


def reference(values):
    result = values.copy()
    with np.errstate(invalid="ignore", over="ignore"):
        for target, policy, mode in zip(TARGETS, POLICIES, MODES, strict=True):
            result += comparison_values(
                cast_reference(values, "f32", target, mode, policy), target
            )
    return result


POLICY = {
    "schema": "t2m.effect-reference.v2",
    "termination": "return",
    "stdout": "",
    "stderr": "",
    "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
}


@pytest.mark.skipif(
    not MERLIN, reason="existing native/shared/Merlin environments required"
)
@pytest.mark.parametrize("shape", [(16,), (2, 16)])
def test_original_descriptor_field_abi_keeps_casts_and_runtime_observations(
    tmp_path, shape
):
    from test_mixed_effects import qualify

    result = convert(tmp_path, shape=shape)
    descriptor, *ordinary = result.trace["descriptor_argument_binding"]["arguments"]
    assert descriptor["descriptor"] and len(descriptor["fields"]) == 3 + 2 * len(shape)
    assert ordinary and descriptor["fields"][1]["binding"]["name"] == "shape0"
    assert (
        sum(row["opcode"] == "tt.fp_to_fp" for row in result.trace["source_operations"])
        == 3
    )
    assert any(
        "tt.tensordesc" in dtype
        for row in result.trace["source_operations"]
        for dtype in row["input_types"]
    )
    options = {
        "block": shape,
        "dimensions": (19,) if len(shape) == 1 else (7, 19),
        "strides": (3,) if len(shape) == 1 else (24, 1),
        "offsets": (-4, 0, 0) if len(shape) == 1 else (-1, 4, 0),
        "capacity": 64 if len(shape) == 1 else 256,
        "padding_nan": True,
        "round_tf32": True,
    }
    values, loaded = arguments("load", "f32", **options)
    capture, qualified = qualify(result, tmp_path, values, POLICY, [reference(loaded)])
    library = _record(qualified / "host/model.so")
    for index, changes in enumerate(
        (
            {"padding_nan": False, "round_tf32": False},
            {"dimensions": (0,) if len(shape) == 1 else (0, 19), "capacity": 0},
            {
                "offsets": (4, 0, 0) if len(shape) == 1 else (2, 4, 0),
                "round_tf32": False,
            },
        )
    ):
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        values, loaded = arguments("load", "f32", **(options | changes))
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(values)}
        )
        np.savez(caller / "reference.npz", out0=reference(loaded))
        (caller / "reference.json").write_text(json.dumps(POLICY))
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
@pytest.mark.parametrize("malformed", ["width", "bounds"])
def test_source_only_preflight_does_not_admit_a_wrong_expanded_descriptor_abi(
    tmp_path, malformed
):
    with pytest.raises(
        ConversionError, match="actual expanded native type|nonnegative runtime bounds"
    ):
        convert(tmp_path, malformed_abi=malformed)
    assert (tmp_path / "translation/expanded-source/lowered.mlir").is_file()


@pytest.mark.skipif(
    not MERLIN, reason="existing native/shared/Merlin environments required"
)
def test_original_descriptor_policy_refuses_before_its_field_expansion(
    tmp_path, monkeypatch
):
    import t2m.shared_tensor as shared

    expand = shared.expand_tensor_pointers
    calls = []

    def checked(source, *args, **kwargs):
        calls.append(source)
        assert len(calls) == 1, (
            "the mixed-policy source reached normalization before validation"
        )
        return expand(source, *args, **kwargs)

    monkeypatch.setattr(shared, "expand_tensor_pointers", checked)
    with pytest.raises(ConversionError, match="explicit valid numeric policy"):
        convert(tmp_path, invalid_policy=True)
    assert len(calls) == 1  # Only the separate, unannotated fixture construction.
