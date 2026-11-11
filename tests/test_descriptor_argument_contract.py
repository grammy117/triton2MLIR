"""Protocol fixtures for field custody; numerical oracles live in integration tests."""

from copy import deepcopy
from dataclasses import replace

import pytest

from t2m.descriptor_arguments import bind_entry, expansions
from t2m.diagnostics import ConversionError
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer


def contract():
    original = {
        "kernel": ["!tt.tensordesc<tensor<2x16xf32>>", "!tt.ptr<f32>", "i32", "i32"]
    }
    normalized = {
        "kernel": [
            "!tt.ptr<f32>",
            "i64",
            "i64",
            "i64",
            "i64",
            "i1",
            "i1",
            "!tt.ptr<f32>",
            "i32",
            "i32",
        ]
    }
    scalars = tuple(
        RuntimeScalar(name, dtype, position, bounds=bounds)
        for name, dtype, position, bounds in (
            ("rows", "i64", 1, (0, 9)),
            ("cols", "i64", 2, (0, 32)),
            ("sr", "i64", 3, (-64, 64)),
            ("sc", "i64", 4, (-8, 8)),
            ("padding", "i1", 5, None),
            ("round", "i1", 6, None),
            ("row", "i32", 8, (-9, 9)),
            ("col", "i32", 9, (-32, 32)),
        )
    )
    abi = TensorABI(
        "kernel",
        (TensorBuffer("x", (256,)), TensorBuffer("out", (32,), "write")),
        runtime_scalars=scalars,
    )
    return original, normalized, abi


@pytest.mark.parametrize(
    "change",
    ("arity", "width", "shape_bounds", "negative_shape", "bool", "allocation_type"),
)
def test_field_binding_refuses_incomplete_or_unproved_interfaces(change):
    original, normalized, abi = contract()
    scalars = list(abi.runtime_scalars)
    if change == "arity":
        scalars.pop()
    elif change == "width":
        scalars[0] = replace(scalars[0], dtype="i32")
    elif change == "shape_bounds":
        scalars[0] = replace(scalars[0], bounds=None)
    elif change == "negative_shape":
        scalars[0] = replace(scalars[0], bounds=(-1, 9))
    elif change == "bool":
        scalars[4] = replace(scalars[4], dtype="i32")
    else:
        abi = replace(
            abi, buffers=(replace(abi.buffers[0], dtype="i64"), abi.buffers[1])
        )
    abi = replace(abi, runtime_scalars=tuple(scalars))
    with pytest.raises(ConversionError, match="descriptor"):
        bind_entry(expansions(original, normalized), abi)


@pytest.mark.parametrize(
    "change", ("arity", "reordered_type", "bool_width", "base_type")
)
def test_actual_native_signature_must_match_the_pinned_expansion(change):
    original, normalized, _ = contract()
    if change == "arity":
        normalized["kernel"].pop()
    elif change == "reordered_type":
        normalized["kernel"][1], normalized["kernel"][5] = (
            normalized["kernel"][5],
            normalized["kernel"][1],
        )
    elif change == "bool_width":
        normalized["kernel"][5] = "i32"
    else:
        normalized["kernel"][0] = "!tt.ptr<i32>"
    with pytest.raises(ConversionError, match="actual|pinned logical field"):
        expansions(original, normalized)


def test_unknown_field_schema_refuses_and_absent_entry_does_not_invent_bindings():
    original, normalized, abi = contract()
    expansion = expansions(original, normalized)
    broken = deepcopy(expansion)
    broken["schema"] = "unknown"
    with pytest.raises(ConversionError, match="unknown logical descriptor"):
        bind_entry(broken, abi)
    assert bind_entry(expansion, replace(abi, entry="ordinary")) is None
