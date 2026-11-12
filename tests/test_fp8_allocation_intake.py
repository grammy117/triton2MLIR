"""Original source policies survive allocation wrappers and descriptor origins."""

import json
import os
from copy import deepcopy

import numpy as np
import pytest
from test_descriptor_state import REQUIRED
from xdsl.dialects.builtin import StringAttr

from t2m.allocation_bindings import (
    PointerArgument,
    ScalarArgument,
    arguments_from_dict,
    convert_allocations,
)
from t2m.descriptor_state_demo import convert as baseline_descriptor
from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.examples.fp8_allocation_reference import update
from t2m.fp8_casts import SOURCE_POLICY
from t2m.frontend import capture_ttir, read_ttir_generic
from t2m.ir import text
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, opcode, parse_shared

pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing native/shared/Merlin environments required"
)


def convert(root, *, descriptor=True, invalid=False):
    from t2m.examples.fp8_allocation_kernels import (
        descriptor_fp8_updates,
        pointer_fp8_updates,
    )

    root.mkdir(parents=True, exist_ok=True)
    if descriptor:
        baseline = baseline_descriptor(
            root / "baseline", dtype="f32", optimizer=os.environ["TRITON_SHARED_OPT"]
        )
        abi = baseline.spec
        _, bindings = arguments_from_dict(
            json.loads(baseline.auxiliary_artifacts["allocation-arguments.json"])
        )
        signature = {
            "first": "tensordesc<fp32[2,16]>",
            "second": "tensordesc<fp32[2,16]>",
            **{name: "i32" for name in ("m", "n", "steps", "capacity")},
        }
        kernel = descriptor_fp8_updates
    else:
        fields = (
            ("n", "i32", (0, 31)),
            ("steps", "i32", (0, 3)),
            ("capacity", "i32", (0, 1024)),
            ("read_origin", "i64", (-8192, 8192)),
            ("write_origin", "i64", (-8192, 8192)),
        )
        abi = TensorABI(
            "pointer_fp8_allocation",
            (TensorBuffer("state", ("capacity",), "readwrite", "f32"),),
            grid=(GridExtent("n", 16),),
            runtime_scalars=tuple(
                RuntimeScalar(name, dtype, index, bounds=bounds)
                for index, (name, dtype, bounds) in enumerate(fields, 1)
            ),
        )
        bindings = (
            PointerArgument("state", "read_origin"),
            PointerArgument("state", "write_origin"),
            *(ScalarArgument(name) for name in ("n", "steps", "capacity")),
        )
        signature = {
            "first": "*fp32",
            "second": "*fp32",
            **{name: "i32" for name in ("n", "steps", "capacity")},
        }
        kernel = pointer_fp8_updates
    source = root / "source.ttir"
    source.write_text(
        capture_ttir(kernel, signature, {}, source_target=("cuda", 90, 32))
    )
    read_ttir_generic(source, root / "fixture-print")
    module = parse_shared((root / "fixture-print/lowered.mlir").read_text())
    casts = [
        operation
        for operation in module.walk()
        if opcode(operation) == "tt.fp_to_fp" and "f8" in str(operation.results[0].type)
    ]
    assert len(casts) == 3
    for cast, policy in zip(
        casts, ("ieee", "satfinite", "satfinite-nan-inf"), strict=True
    ):
        cast.attributes[SOURCE_POLICY] = StringAttr("unknown" if invalid else policy)
    source.write_text(text(module))
    return convert_allocations(
        source,
        abi,
        bindings,
        root / "binding",
        source_entry=kernel.__name__,
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer",
        address_policy="source-words",
        fp8_overflow="per-cast",
    )


def arguments(
    *,
    m=2,
    n=13,
    steps=2,
    capacity=256,
    read_origin=8,
    write_origin=16,
    read_row=24,
    read_col=1,
    write_row=24,
    write_col=1,
):
    words = np.asarray(
        [
            0,
            0x80000000,
            0x3F880000,
            0x3F900000,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0x7F812345,
            0x47800000,
            0xC7800000,
            0x00000001,
        ],
        np.uint32,
    )
    state = np.resize(words, capacity).view(np.float32).copy()
    expected = update(
        state,
        m,
        n,
        steps,
        capacity,
        read_origin,
        write_origin,
        read_row,
        read_col,
        write_row,
        write_col,
    )
    values = [
        state,
        *(np.asarray(value, np.int32) for value in (m, n, steps, capacity)),
        *(
            np.asarray(value, np.int64)
            for value in (
                read_origin,
                write_origin,
                read_row,
                read_col,
                write_row,
                write_col,
            )
        ),
    ]
    return values, expected


POLICY = {
    "schema": "t2m.effect-reference.v2",
    "termination": "return",
    "stdout": "",
    "stderr": "",
    "results": {"archive": "reference.npz", "comparisons": [{"kind": "ieee"}]},
}


@pytest.mark.parametrize("descriptor", [False, True], ids=["pointer", "descriptor"])
def test_original_source_casts_survive_allocation_normalization_and_execution(
    tmp_path, descriptor
):
    from test_mixed_effects import qualify

    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.source_capsule import record, validate

    result = convert(tmp_path, descriptor=descriptor)
    capsule = result.trace["source_argument_binding"]["source_capsule"]
    assert (
        capsule["schema"] == "t2m.source-allocation-capsule.v2"
        and capsule["normalized"] == descriptor
    )
    protection = json.loads(result.auxiliary_artifacts["allocation-protection.json"])
    assert len(protection["fp8_conversions"]) == 3
    assert (
        sum(
            op["opcode"] == "tt.fp_to_fp"
            for op in protection["source_graph"]["operations"]
        )
        == 6
    )

    def inputs(**changes):
        if not descriptor:
            changes["m"] = 1
            if not changes.get("capacity", 256):
                changes["n"] = 0 if changes.get("steps", 2) else changes.get("n", 13)
        values, expected = arguments(**changes)
        return (
            values if descriptor else [values[i] for i in (0, 2, 3, 4, 5, 6)]
        ), expected

    values, expected = inputs()
    capture, qualified = qualify(result, tmp_path, values, POLICY, [expected])
    library = _record(qualified / "host/model.so")
    for index, changes in enumerate(
        (
            {
                "m": 5,
                "n": 19 if descriptor else 31,
                "steps": 3,
                "read_origin": 8,
                "write_origin": 8,
            },
            {
                "steps": 1,
                "read_origin": 48,
                "write_origin": 64,
                "read_row": -24,
                "write_row": -24,
            },
            {"m": 0, "capacity": 0, "read_origin": 8191, "write_origin": 8191},
            {
                "m": 5,
                "n": 19,
                "steps": 0,
                "capacity": 0,
                "read_origin": 8191,
                "write_origin": 8191,
            },
        )
    ):
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        values, expected = inputs(**changes)
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(values)}
        )
        np.savez(caller / "reference.npz", out0=expected)
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
    for change in (
        "generic",
        "protected",
        "original",
        "entry",
        "authority",
        "evaluator",
        "printer",
        "intake",
        "downgrade",
    ):
        trace, artifacts = deepcopy(result.trace), dict(result.auxiliary_artifacts)
        report = json.loads(artifacts["allocation-protection.json"])
        if change in ("generic", "protected", "original"):
            report["artifacts"][change] = record("changed source")
        elif change == "entry":
            report["source_entry"] = "other"
        elif change == "authority":
            report["grants_semantics"] = True
        elif change == "evaluator":
            report["reference_sha256"] = "0" * 64
        elif change == "printer":
            native = json.loads(artifacts["allocation-native-print.json"])
            native["native_generic_roundtrip_verified"] = False
            artifacts["allocation-native-print.json"] = json.dumps(native)
        elif change == "intake":
            name = (
                "allocation-intake.json"
                if descriptor
                else "allocation-source-stage.json"
            )
            intake = json.loads(artifacts[name])
            if descriptor:
                intake["source_sha256"] = record(artifacts["allocation-original.ttir"])[
                    "sha256"
                ]
            else:
                intake["artifacts"]["source.ttir"] = record(
                    artifacts["allocation-original.ttir"]
                )
            artifacts[name] = json.dumps(intake)
        else:
            trace["source_argument_binding"]["source_capsule"]["schema"] = (
                "t2m.source-allocation-capsule.v1"
            )
        artifacts["allocation-protection.json"] = json.dumps(report)
        pins = trace["source_argument_binding"]["source_capsule"]["artifacts"]
        for name in pins:
            pins[name] = record(artifacts[name])
        with pytest.raises(ValueError, match="source|protection|roster"):
            validate(trace, artifacts, result.source_ttir)
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_effect as target

        options = {
            "merlin_root": os.environ["T2M_MERLIN_ROOT"],
            "python": os.environ["T2M_MERLIN_PYTHON"],
        }
        target.qualify(capture, tmp_path / "target", **options)
        target.verify(tmp_path / "target", **options)


@pytest.mark.parametrize("descriptor", [False, True], ids=["pointer", "descriptor"])
def test_original_invalid_policy_cannot_reach_allocation_normalization(
    tmp_path, monkeypatch, descriptor
):
    from t2m import frontend

    expand = frontend.expand_tensor_pointers
    calls = []

    def checked(*args, **kwargs):
        calls.append(args[0])
        assert len(calls) <= int(descriptor), (
            "invalid original policy reached allocation normalization"
        )
        return expand(*args, **kwargs)

    monkeypatch.setattr(frontend, "expand_tensor_pointers", checked)
    with pytest.raises(ConversionError, match="explicit valid numeric policy"):
        convert(tmp_path, descriptor=descriptor, invalid=True)
    assert len(calls) == int(descriptor)  # The independent legacy fixture only.
