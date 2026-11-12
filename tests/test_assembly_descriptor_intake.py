"""Decoded source arithmetic crosses original descriptor normalization."""

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_descriptor_state import REQUIRED
from test_source_assembly import PROVIDER

from t2m.allocation_bindings import (
    ScalarArgument,
    arguments_from_dict,
    convert_allocations,
)
from t2m.descriptor_state_demo import convert as baseline_descriptor
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar

pytestmark = pytest.mark.skipif(
    not (REQUIRED and PROVIDER),
    reason="existing native/Merlin and source provider required",
)


def convert(root, *, dtype="i64", words=False, assembly=None):
    from t2m.examples.assembly_descriptor_kernels import (
        assembly_descriptor_state,
        assembly_descriptor_words,
    )

    baseline = baseline_descriptor(
        root / "baseline", dtype="i64", optimizer=os.environ["TRITON_SHARED_OPT"]
    )
    _, bindings = arguments_from_dict(
        json.loads(baseline.auxiliary_artifacts["allocation-arguments.json"])
    )
    abi = replace(
        baseline.spec, buffers=(replace(baseline.spec.buffers[0], dtype=dtype),)
    )
    if words:
        spec = importlib.util.spec_from_file_location(
            "assembly_word_example",
            Path(__file__).resolve().parents[1]
            / "adapters/ptx/examples/model_reference.py",
        )
        sample = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sample)
        asm = sample.program(64 if dtype == "i64" else 32)
        abi = replace(
            abi,
            runtime_scalars=(
                *abi.runtime_scalars,
                RuntimeScalar(
                    "shift",
                    "i32",
                    len(abi.source_bindings),
                    bounds=(-(1 << 31), (1 << 31) - 1),
                ),
            ),
        )
        bindings += (ScalarArgument("shift"),)
    else:
        asm = f"add.u{64 if dtype == 'i64' else 32} $0, $1, 1;"
    if assembly is not None:
        asm = assembly
    kernel = assembly_descriptor_words if words else assembly_descriptor_state
    source = root / "source.ttir"
    source.write_text(
        capture_ttir(
            kernel,
            {
                "first": f"tensordesc<{dtype}[2,16]>",
                "second": f"tensordesc<{dtype}[2,16]>",
                **{
                    name: "i32"
                    for name in (
                        "m",
                        "n",
                        "steps",
                        "capacity",
                        *(["shift"] if words else []),
                    )
                },
                "ASM": "constexpr",
                "WORDS64": "constexpr",
            },
            {"ASM": asm, "WORDS64": dtype == "i64"},
        )
    )
    return convert_allocations(
        source,
        abi,
        bindings,
        root / "binding",
        source_entry=kernel.__name__,
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer",
        address_policy="source-words",
        assembly_provider="ptx-integer-v1",
    )


@pytest.mark.parametrize("dtype", ["i32", "i64"])
@pytest.mark.parametrize("words", [False, True], ids=["add", "tuple-shifts"])
def test_tensor_source_assembly_precedes_descriptor_normalization_and_execution(
    tmp_path, dtype, words
):
    from test_mixed_effects import qualify

    from t2m.examples.assembly_descriptor_reference import arguments
    from t2m.examples.descriptor_reference import policy

    result = convert(tmp_path, dtype=dtype, words=words)
    from t2m.shared_tensor import opcode, parse_shared

    protection = json.loads(result.auxiliary_artifacts["allocation-protection.json"])
    assert len(protection["assembly_conversions"]) == 1
    assert (
        protection["assembly_conversions"][0]["expansion"]
        == "standard-elementwise-arith"
    )
    protected = parse_shared(result.auxiliary_artifacts["allocation-protected.ttir"])
    assert all(
        opcode(op).split(".")[0] not in ("linalg", "tensor") for op in protected.walk()
    )
    values, expected = arguments(dtype, words=words)
    capture, qualified = qualify(result, tmp_path, values, policy(), [expected])
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    library = _record(qualified / "host/model.so")
    cases = [
        {"m": 5, "n": 19, "read_origin": 8, "write_origin": 8},
        {
            "steps": 1,
            "read_origin": 48,
            "write_origin": 64,
            "read_row": -24,
            "write_row": -24,
        },
        {"m": 0, "capacity": 0, "read_origin": 8191, "write_origin": 8191},
        {"steps": 0, "capacity": 0, "read_origin": 8191, "write_origin": 8191},
    ]
    if words:
        cases.extend({"shift": count} for count in (-1, 32, 33, 64, 65))
    for index, changes in enumerate(cases):
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        values, expected = arguments(dtype, words=words, **changes)
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(values)}
        )
        np.savez(caller / "reference.npz", out0=expected)
        (caller / "reference.json").write_text(json.dumps(policy()))
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
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_effect as target

        options = {
            "merlin_root": os.environ["T2M_MERLIN_ROOT"],
            "python": os.environ["T2M_MERLIN_PYTHON"],
        }
        target.qualify(capture, tmp_path / "target", **options)
        target.verify(tmp_path / "target", **options)


@pytest.mark.parametrize("dtype", ["i32", "i64"])
def test_direct_original_descriptor_and_assembly_reach_the_field_interface(
    tmp_path, dtype
):
    from test_mixed_effects import qualify
    from xdsl.dialects import builtin

    from t2m.examples.descriptor_reference import policy
    from t2m.examples.external_descriptor_reference import arguments
    from t2m.external_descriptor_demo import convert as baseline
    from t2m.frontend import read_ttir_generic
    from t2m.ir import text
    from t2m.shared_tensor import convert_shared, opcode, parse_shared

    ordinary = baseline(
        tmp_path / "baseline",
        kind="load",
        dtype=dtype,
        optimizer=os.environ["TRITON_SHARED_OPT"],
    )
    source = tmp_path / "baseline/source.ttir"
    read_ttir_generic(source, tmp_path / "fixture-print")
    module = parse_shared((tmp_path / "fixture-print/lowered.mlir").read_text())
    (loaded,) = [op for op in module.walk() if opcode(op) == "tt.descriptor_load"]
    (value,) = loaded.results
    users = tuple(value.uses)
    asm = builtin.UnregisteredOp.with_name("tt.elementwise_inline_asm").create(
        operands=[value],
        result_types=[value.type],
        properties={
            "asm_string": builtin.StringAttr(
                f"add.u{32 if dtype == 'i32' else 64} $0, $1, 1;"
            ),
            "constraints": builtin.StringAttr("=r,r" if dtype == "i32" else "=l,l"),
            "pure": builtin.IntegerAttr.from_bool(True),
            "packed_element": builtin.IntegerAttr(1, builtin.i32),
        },
    )
    loaded.parent.insert_op_after(asm, loaded)
    for use in users:
        use.operation.operands[use.index] = asm.results[0]
    original = tmp_path / "source.ttir"
    original.write_text(text(module))
    result = convert_shared(
        original,
        ordinary.spec,
        tmp_path / "translation",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer",
        address_policy="source-words",
        assembly_provider="ptx-integer-v1",
    )
    (report,) = [
        row
        for row in result.trace["transformations"]
        if row.get("kind") == "source-assembly-implementation"
    ]
    assert (
        report["source_graph"] == "original"
        and report["expansion"] == "standard-elementwise-arith"
    )
    values, loaded = arguments(
        "load",
        dtype,
        block=(2, 16),
        dimensions=(7, 19),
        strides=(24, 1),
        offsets=(-1, 4, 0),
        capacity=256,
    )
    expected = loaded + np.asarray(1, loaded.dtype)
    capture, _ = qualify(result, tmp_path, values, policy(), [expected])
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_effect as target

        options = {
            "merlin_root": os.environ["T2M_MERLIN_ROOT"],
            "python": os.environ["T2M_MERLIN_PYTHON"],
        }
        target.qualify(capture, tmp_path / "target", **options)
        target.verify(tmp_path / "target", **options)


@pytest.mark.parametrize("provider", [None, "ptx-integer-v1"])
def test_unimplemented_descriptor_source_refuses_before_any_native_normalization(
    tmp_path, monkeypatch, provider
):
    from t2m import frontend
    from t2m.diagnostics import ConversionError

    # Construct source without admitting it or invoking a native transformation.
    from t2m.examples.assembly_descriptor_kernels import assembly_descriptor_state
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = tmp_path / "source.ttir"
    source.write_text(
        capture_ttir(
            assembly_descriptor_state,
            {
                "first": "tensordesc<i64[2,16]>",
                "second": "tensordesc<i64[2,16]>",
                **{name: "i32" for name in ("m", "n", "steps", "capacity")},
                "ASM": "constexpr",
                "WORDS64": "constexpr",
            },
            {"ASM": "mov.u64 $0, %clock64;", "WORDS64": True},
        )
    )
    abi = TensorABI(
        "assembly_descriptor_state",
        (TensorBuffer("state", (256,), "readwrite", "i64"),),
    )

    def forbidden(*_a, **_kw):
        pytest.fail("unimplemented original source reached native normalization")

    monkeypatch.setattr(frontend, "expand_tensor_pointers", forbidden)
    monkeypatch.setattr("t2m.shared_tensor.expand_tensor_pointers", forbidden)
    with pytest.raises(
        ConversionError, match="qualified effect|hardware|register|operand|instruction"
    ):
        convert_shared(
            source,
            abi,
            tmp_path / "translation",
            optimizer=os.environ["TRITON_SHARED_OPT"],
            memory_profile="guarded-buffer",
            address_policy="source-words",
            assembly_provider=provider,
        )
