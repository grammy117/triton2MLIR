"""Representation planning is source-only, bounded and fail-closed."""

import hashlib
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import pytest

from t2m.auto_memory import convert
from t2m.diagnostics import ConversionError
from t2m.memory_contract import GUARDED_PROFILE, TENSOR_PROFILE, WORD_PROFILE
from t2m.memory_selection import validate_selection
from t2m.shared_tensor import TensorABI, TensorBuffer


def request(
    tmp_path, monkeypatch, *, admitted=0, exception=None, mutate=False, **overrides
):
    from t2m import shared_tensor

    source = tmp_path / "source.ttir"
    source.write_text("source-only policy unit fixture")
    abi = TensorABI(
        "entry", (TensorBuffer("x", (4,)), TensorBuffer("out", (4,), "write"))
    )
    calls = []
    expected = [
        ("tensor", "structured"),
        ("guarded-buffer", "structured"),
        ("guarded-buffer", "source-words"),
    ]

    def convert_candidate(snapshot, actual_abi, work, **options):
        assert snapshot.read_bytes() == b"source-only policy unit fixture"
        assert actual_abi == abi
        assert "inputs" not in options and "reference" not in options
        calls.append(options)
        if mutate:
            source.write_text("changed source")
        if exception is not None:
            raise exception
        if admitted is None or len(calls) <= admitted:
            raise ConversionError(
                "unproved memory representation",
                source_id="ttir.op.0007",
                opcode="tt.store",
            )
        pair = options["memory_profile"], options["address_policy"]
        profile = {
            expected[0]: TENSOR_PROFILE,
            expected[1]: GUARDED_PROFILE,
            expected[2]: WORD_PROFILE,
        }[pair]
        return SimpleNamespace(
            trace={
                "source_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "profile": profile,
            }
        )

    monkeypatch.setattr(shared_tensor, "convert_shared", convert_candidate)
    options = {
        "optimizer": None,
        "address_policy": "structured",
        "effect_profile": "ordinary",
        "f32_atomic_add": None,
        "fp8_overflow": None,
        "byte_order": None,
        "assembly_provider": None,
        "source_layout": None,
    }
    options.update(overrides)
    return source, abi, tmp_path / "selection", options, calls


@pytest.mark.parametrize("admitted", [0, 1, 2])
def test_first_admitted_representation_retains_all_prior_diagnostics(
    tmp_path, monkeypatch, admitted
):
    source, abi, work, options, calls = request(
        tmp_path, monkeypatch, admitted=admitted
    )
    result = convert(source, abi, work, **options)
    assert [(row["memory_profile"], row["address_policy"]) for row in calls] == [
        ("tensor", "structured"),
        ("guarded-buffer", "structured"),
        ("guarded-buffer", "source-words"),
    ][: admitted + 1]
    record = json.loads((work / "memory-selection.json").read_text())
    assert record == result.trace["memory_selection"]
    assert [row["status"] for row in record["attempts"]] == [
        "not-admitted"
    ] * admitted + ["admitted"]
    assert all(
        row["source_id"] == "ttir.op.0007" and row["opcode"] == "tt.store"
        for row in record["attempts"][:-1]
    )
    validate_selection(result.trace)


@pytest.mark.parametrize(
    "overrides",
    [
        {"address_policy": "source-words"},
        {"byte_order": "big"},
        {"effect_profile": "atomics"},
        {"effect_profile": "typed-atomics", "f32_atomic_add": "ieee"},
    ],
)
def test_explicit_policy_restrictions_never_try_an_incompatible_representation(
    tmp_path, monkeypatch, overrides
):
    source, abi, work, options, calls = request(tmp_path, monkeypatch, **overrides)
    result = convert(source, abi, work, **options)
    assert len(calls) == 1
    assert (calls[0]["memory_profile"], calls[0]["address_policy"]) == (
        "guarded-buffer",
        "source-words",
    )
    for name in ("effect_profile", "f32_atomic_add", "fp8_overflow", "byte_order"):
        assert calls[0][name] == options[name]
    validate_selection(result.trace)


def test_all_refused_candidates_leave_located_diagnostics_and_no_result(
    tmp_path, monkeypatch
):
    source, abi, work, options, calls = request(tmp_path, monkeypatch, admitted=None)
    with pytest.raises(ConversionError) as caught:
        convert(source, abi, work, **options)
    assert (
        caught.value.source_id == "ttir.op.0007" and caught.value.opcode == "tt.store"
    )
    assert len(calls) == 3
    record = json.loads((work / "memory-selection.json").read_text())
    assert record["status"] == "not-admitted"
    assert len(record["attempts"]) == 3 and all(
        row["status"] == "not-admitted" for row in record["attempts"]
    )


def test_unexpected_failure_is_recorded_and_propagated_without_retry(
    tmp_path, monkeypatch
):
    source, abi, work, options, calls = request(
        tmp_path, monkeypatch, exception=RuntimeError("compiler failure")
    )
    with pytest.raises(RuntimeError, match="compiler failure"):
        convert(source, abi, work, **options)
    assert len(calls) == 1
    row = json.loads((work / "memory-selection.json").read_text())["attempts"][0]
    assert row["status"] == "failed" and row["exception"] == "RuntimeError"


@pytest.mark.parametrize("admitted", [0, None])
def test_changed_input_cannot_select_a_representation_for_another_source(
    tmp_path, monkeypatch, admitted
):
    source, abi, work, options, calls = request(
        tmp_path, monkeypatch, admitted=admitted, mutate=True
    )
    with pytest.raises(ConversionError, match="source changed"):
        convert(source, abi, work, **options)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "damage", ["source", "profile", "order", "diagnostic", "policies"]
)
def test_rehashed_selection_metadata_cannot_change_actual_profile_or_admission_history(
    tmp_path, monkeypatch, damage
):
    source, abi, work, options, _calls = request(tmp_path, monkeypatch, admitted=2)
    trace = deepcopy(convert(source, abi, work, **options).trace)
    record = trace["memory_selection"]
    if damage == "source":
        record["source_sha256"] = "0" * 64
    elif damage == "profile":
        trace["profile"] = TENSOR_PROFILE
    elif damage == "order":
        record["attempts"].reverse()
    elif damage == "diagnostic":
        record["attempts"][0]["diagnostic"] = ""
    else:
        record["requested"]["effect_profile"] = "atomics"
    with pytest.raises(ValueError):
        validate_selection(trace)


@pytest.mark.parametrize(
    "overrides",
    [
        {"effect_profile": "unknown"},
        {"byte_order": "unknown"},
        {"fp8_overflow": "unknown"},
        {"f32_atomic_add": "ieee"},
    ],
)
def test_invalid_numeric_effect_policies_fail_before_any_candidate(
    tmp_path, monkeypatch, overrides
):
    source, abi, work, options, calls = request(tmp_path, monkeypatch, **overrides)
    with pytest.raises(ValueError):
        convert(source, abi, work, **options)
    assert not calls and not work.exists()


@pytest.mark.skipif(
    not os.environ.get("TRITON_SHARED_OPT"), reason="existing shared optimizer required"
)
def test_real_unselected_assembly_is_not_erased_by_another_memory_representation(
    tmp_path,
):
    from test_source_assembly import source

    from t2m.shared_tensor import convert_shared

    path, abi = source(tmp_path)
    work = tmp_path / "auto"
    with pytest.raises(ConversionError) as caught:
        convert_shared(
            path,
            abi,
            work,
            optimizer=os.environ["TRITON_SHARED_OPT"],
            memory_profile="auto",
        )
    assert caught.value.opcode == "tt.elementwise_inline_asm" and caught.value.source_id
    report = json.loads((work / "memory-selection.json").read_text())
    assert len(report["attempts"]) == 3
    assert all(
        row["status"] == "not-admitted" and row["opcode"] == "tt.elementwise_inline_asm"
        for row in report["attempts"]
    )


@pytest.mark.skipif(
    not all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON")),
    reason="existing shared/native toolchains required",
)
def test_real_pointwise_source_uses_tensor_form_and_keeps_complete_native_result(tmp_path):
    import numpy as np
    from test_shared_tensor import source_case

    from t2m.native import execute_native
    from t2m.shared_tensor import convert_shared

    source, abi = source_case("matrix_add", tmp_path)
    result = convert_shared(
        source, abi, tmp_path / "auto", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="auto",
    )
    assert result.trace["memory_selection"]["attempts"] == [
        {"memory_profile": "tensor", "address_policy": "structured", "status": "admitted"}
    ]
    assert not any(op.name.startswith(("memref.", "gpu.", "tt.")) for op in result.module.walk())
    x = np.arange(32, dtype=np.float32).reshape(4, 8)
    y = np.flip(x, axis=1).copy() * np.float32(-2)
    arrays = [x, y, np.full_like(x, 719)]
    before = [value.copy() for value in arrays]
    (actual,), report = execute_native(
        result.mlir_text, arrays, tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"],
    )
    np.testing.assert_array_equal(actual, x + y)
    for actual_input, original in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(actual_input, original)
    assert report["llvm_lowering_verified"]
