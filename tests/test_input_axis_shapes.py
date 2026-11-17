"""Runtime dimensions come from declared input axes without invented scalars."""

import os
from copy import deepcopy

import numpy as np
import pytest

from t2m.runtime_shapes import resolve_shapes


def contract():
    return {"schema": "t2m.runtime-shapes.v2", "inputs": [{"shape": ["rows", 3], "dtype": "i32"}],
        "outputs": [{"shape": ["rows", 3], "dtype": "i32"}],
        "dimension_bindings": {"rows": {"argument": 0, "axis": 0, "bounds": [0, 31]}}}


@pytest.mark.parametrize("prefix", ["t2m", "merlin"])
@pytest.mark.parametrize("rows", [0, 1, 7, 31])
def test_axis_dimensions_resolve_empty_and_runtime_results(prefix, rows):
    description = contract()
    description["schema"] = prefix + ".runtime-shapes.v2"
    assert resolve_shapes(description, [np.zeros((rows, 3), np.int32)]) == ((rows, 3),)


@pytest.mark.parametrize("damage", ["downgrade", "missing", "axis", "argument", "static-axis", "bounds",
                                    "null-bounds", "short-shape", "unknown", "output"])
def test_input_axis_claims_cannot_replace_actual_axes_or_bounds(damage):
    description = contract()
    binding = description["dimension_bindings"]["rows"]
    if damage == "downgrade": description["schema"] = "t2m.runtime-shapes.v1"
    elif damage == "missing": description.pop("dimension_bindings")
    elif damage == "axis": binding["axis"] = -1
    elif damage == "argument": binding["argument"] = 1
    elif damage == "static-axis": binding["axis"] = 1
    elif damage == "bounds": binding["bounds"] = [8, 31]
    elif damage == "null-bounds": binding["bounds"] = None
    elif damage == "short-shape": description["inputs"][0]["shape"] = []
    elif damage == "unknown": binding["authority"] = "shape-proof"
    else: description["outputs"][0]["shape"][0] = "unbound"
    with pytest.raises(ValueError):
        resolve_shapes(description, [np.zeros((7, 3), np.int32)])


def test_shared_axis_names_check_every_input_and_do_not_override_scalar_bindings():
    description = contract()
    description["inputs"].append(deepcopy(description["inputs"][0]))
    with pytest.raises(ValueError):
        resolve_shapes(description, [np.zeros((7, 3), np.int32), np.zeros((8, 3), np.int32)])
    description["inputs"].append({"shape": [], "dtype": "i32", "scalar": "rows"})
    with pytest.raises(ValueError):
        resolve_shapes(description, [np.zeros((7, 3), np.int32), np.zeros((7, 3), np.int32), np.asarray(7, np.int32)])


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native toolchain required")
def test_native_dynamic_result_allocation_uses_input_axes_and_original_public_abi(tmp_path):
    from t2m.native import execute_native

    source = '''module {func.func @forward(%x:tensor<?x3xi32>)->tensor<?x3xi32> {
      return %x:tensor<?x3xi32>
    }}'''
    for rows in (0, 1, 7, 31):
        x = np.arange(rows * 3, dtype=np.int32).reshape(rows, 3)
        (actual,), _ = execute_native(source, [x], tmp_path / f"shape-{rows}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=contract())
        assert (actual.shape, actual.dtype, actual.tobytes()) == (x.shape, x.dtype, x.tobytes())


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native toolchain required")
@pytest.mark.parametrize("side", ["inputs", "outputs"])
def test_native_axis_contract_dtype_cannot_override_the_actual_program_type(tmp_path, side):
    from t2m.native import execute_native

    source = '''module {func.func @forward(%x:tensor<?x3xi32>)->tensor<?x3xi32> {
      return %x:tensor<?x3xi32>
    }}'''
    declared = contract()
    declared[side][0]["dtype"] = "f32"
    with pytest.raises(ValueError, match="typed.*dtype"):
        execute_native(source, [np.zeros((7, 3), np.int32)], tmp_path / "invalid",
                       python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=declared)
