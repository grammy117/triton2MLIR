"""Typed descriptor bindings preserve original argument order and widths."""

from dataclasses import replace

import pytest

from t2m.allocation_bindings import (
    DescriptorArgument,
    PointerArgument,
    ScalarArgument,
    arguments_as_dict,
    arguments_from_dict,
)
from t2m.descriptor_arguments import expansions, has_parameters
from t2m.descriptor_bindings import ExtendScalar, expand
from t2m.diagnostics import ConversionError
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer


def case():
    original = {"source": ["!tt.tensordesc<tensor<2x16xf32>>", "i32"]}
    native = {"source": ["!tt.ptr<f32>", *(["i64"] * 4), "i1", "i1", "i32"]}
    abi = TensorABI(
        "bound",
        (TensorBuffer("state", (256,), "readwrite"),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 1, bounds=(0, 16)),
            RuntimeScalar("origin", "i64", 2, bounds=(-128, 128)),
            RuntimeScalar("padding", "i1", 3),
        ),
    )
    bindings = (
        DescriptorArgument("state", (2, "n"), (16, 1), "origin", "padding"),
        ScalarArgument("n"),
    )
    return original, expansions(original, native), abi, bindings


def test_original_descriptor_configuration_roundtrips_and_extends_shape_in_ir():
    original, native, abi, bindings = case()
    configuration = arguments_as_dict("source", bindings)
    assert configuration["schema"] == "t2m.source-allocation-arguments.v2"
    assert arguments_from_dict(configuration) == ("source", bindings)
    flat, report = expand(bindings, native, "source", abi)
    assert flat == (
        PointerArgument("state", "origin"),
        2,
        ExtendScalar("n"),
        16,
        1,
        ScalarArgument("padding"),
        0,
        ScalarArgument("n"),
    )
    assert report["arguments"][0]["source_argument"] == 0
    assert report["grants_semantics"] is False
    assert has_parameters(original)
    assert not has_parameters({"ordinary": ["!tt.ptr<f32>", "i32"]})


@pytest.mark.parametrize(
    "change",
    (
        "rank",
        "arity",
        "ordinary",
        "shape_bounds",
        "negative_shape",
        "width",
        "flag",
        "missing_scalar",
        "no_native",
    ),
)
def test_invalid_descriptor_interfaces_refuse_before_binding(change):
    _, native, abi, bindings = case()
    descriptor = bindings[0]
    if change == "rank":
        descriptor = replace(descriptor, shape=(2,), strides=(1,))
    elif change == "arity":
        bindings = bindings[:-1]
    elif change == "ordinary":
        bindings = (descriptor, descriptor)
    elif change in ("shape_bounds", "negative_shape", "width"):
        scalar = replace(
            abi.runtime_scalars[0],
            **(
                {"bounds": None}
                if change == "shape_bounds"
                else {"bounds": (-1, 16)}
                if change == "negative_shape"
                else {"dtype": "i16"}
            ),
        )
        abi = replace(abi, runtime_scalars=(scalar, *abi.runtime_scalars[1:]))
    elif change == "flag":
        descriptor = replace(descriptor, padding_nan="n")
    elif change == "missing_scalar":
        descriptor = replace(descriptor, strides=("unknown", 1))
    else:
        native = None
    bindings = (descriptor, *bindings[1:])
    with pytest.raises(ConversionError, match="descriptor|roster"):
        expand(bindings, native, "source", abi)


@pytest.mark.parametrize(
    "fields",
    (
        {"shape": ()},
        {"shape": (-1, 16)},
        {"strides": (1,)},
        {"offset": True},
        {"offset": 2**63},
        {"padding_nan": 1},
        {"round_f32_to_tf32": "bad name"},
        {"shape": (True, 16)},
    ),
)
def test_invalid_descriptor_values_refuse(fields):
    with pytest.raises(ValueError):
        DescriptorArgument(
            **{"allocation": "state", "shape": (2, 16), "strides": (16, 1), **fields}
        )


def test_saved_descriptor_schema_and_unknown_fields_are_strict():
    _, _, _, bindings = case()
    configuration = arguments_as_dict("source", bindings)
    for bad in (
        {**configuration, "schema": "t2m.source-allocation-arguments.v1"},
        {
            **configuration,
            "arguments": [{**configuration["arguments"][0], "ignored": True}],
        },
        {**configuration, "arguments": [{"kind": "scalar", "name": "n"}]},
    ):
        with pytest.raises(ValueError):
            arguments_from_dict(bad)
