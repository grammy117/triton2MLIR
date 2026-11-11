"""Storage metadata cannot alter the public ABI or understate guard capacity."""

import copy
import json

import pytest
from xdsl.dialects import builtin, func

from t2m.typed_storage_contract import checked_guard_costs, storage_spec


def fixture():
    public = {
        "buffers": [
            {"name": "state", "dtype": "f32", "shape": ["n", 3], "effect": "readwrite"}
        ]
    }
    profile = {
        "schema": "t2m.typed-storage-transport.v1",
        "byte_order": "big",
        "grants_semantics": False,
        "roots": [
            {
                "name": "state",
                "logical_dtype": "f32",
                "logical_shape": ["n", 3],
                "storage_dtype": "i8",
                "storage_shape": ["n", 3, 4],
                "element_bytes": 4,
                "effect": "readwrite",
            }
        ],
    }
    trace = {
        "spec": public,
        "profile": "shared-source-word-buffer",
        "typed_storage_transport": profile,
    }
    entry = func.FuncOp("test", ([], []))
    entry.attributes.update(
        {
            "prov.typed_storage_transport": builtin.StringAttr(json.dumps(profile)),
            "prov.memory_guard_bytes_per_result_element": builtin.IntegerAttr(
                64, builtin.i64
            ),
            "prov.memory_guard_bytes_by_result_element": builtin.DenseArrayBase.from_list(
                builtin.i64, [64]
            ),
        }
    )
    return trace, entry


def test_storage_is_derived_without_changing_public_shapes_or_types():
    trace, entry = fixture()
    original = copy.deepcopy(trace)
    internal = storage_spec(trace)
    assert internal["buffers"][0] == {
        "name": "state",
        "dtype": "i8",
        "shape": ["n", 3, 4],
        "effect": "readwrite",
    }
    assert checked_guard_costs(entry, trace, 16) == [64]
    assert trace == original


@pytest.mark.parametrize(
    "change",
    (
        "width",
        "shape",
        "type",
        "effect",
        "root",
        "duplicate",
        "order",
        "grant",
        "profile",
    ),
)
def test_inconsistent_transport_declarations_refuse(change):
    trace, _ = fixture()
    profile = trace["typed_storage_transport"]
    root = profile["roots"][0]
    if change == "width":
        root["element_bytes"] = 8
    if change == "shape":
        root["storage_shape"] = ["n", 12]
    if change == "type":
        root["logical_dtype"] = "i32"
    if change == "effect":
        root["effect"] = "read"
    if change == "root":
        root["name"] = "other"
    if change == "duplicate":
        profile["roots"].append(copy.deepcopy(root))
    if change == "order":
        profile["byte_order"] = "native"
    if change == "grant":
        profile["grants_semantics"] = True
    if change == "profile":
        trace["profile"] = "shared-functional-tensor"
    with pytest.raises(ValueError, match="typed storage"):
        storage_spec(trace)


@pytest.mark.parametrize(
    "change", ("maximum", "vector", "marker", "missing-profile", "missing-vector")
)
def test_guard_contract_cannot_understate_byte_storage(change):
    trace, entry = fixture()
    if change == "maximum":
        entry.attributes["prov.memory_guard_bytes_per_result_element"] = (
            builtin.IntegerAttr(16, builtin.i64)
        )
    if change == "vector":
        entry.attributes["prov.memory_guard_bytes_by_result_element"] = (
            builtin.DenseArrayBase.from_list(builtin.i64, [16])
        )
    if change == "marker":
        entry.attributes["prov.typed_storage_transport"] = builtin.StringAttr("{}")
    if change == "missing-profile":
        trace.pop("typed_storage_transport")
    if change == "missing-vector":
        entry.attributes.pop("prov.memory_guard_bytes_by_result_element")
    with pytest.raises(ValueError, match="scratch storage"):
        checked_guard_costs(entry, trace, 16)
