"""Source parameters keep their element units on differently typed owning roots."""

import json

import pytest
from test_descriptor_state import REQUIRED

from t2m.examples.typed_parameter_reference import arguments

pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


@pytest.mark.parametrize("dtype", ("i16", "f16", "bf16", "i64", "f64"))
@pytest.mark.parametrize("order", ("little", "big"))
@pytest.mark.parametrize("descriptor", (False, True))
def test_changed_width_source_parameters_share_the_typed_root(
    tmp_path, dtype, order, descriptor
):
    from test_indexed_descriptors import replay
    from test_mixed_effects import qualify

    from t2m.examples.descriptor_reference import policy

    if descriptor:
        from t2m.descriptor_state_demo import convert

        result = convert(
            tmp_path / "source",
            dtype="i32",
            word_views=True,
            root_dtype=dtype,
            byte_order=order,
        )
    else:
        from t2m.allocation_word_demo import convert

        result = convert(tmp_path / "source", root_dtype=dtype, byte_order=order)
    assert result.spec.buffers[0].dtype == dtype
    binding = result.trace["source_argument_binding"]
    assert (
        sum(
            r.get("representation") == "typed-root-byte-view-v1"
            for r in binding["bindings"]
        )
        == 2
    )
    assert result.trace["typed_storage_transport"]["roots"][0]["logical_dtype"] == dtype
    values, expected = arguments(dtype, order, descriptor=descriptor)
    _, qualified = qualify(result, tmp_path, values, policy(), [expected])
    for index, options in enumerate(
        (
            {
                "n": 19 if descriptor else 31,
                "m": 5,
                "steps": 3,
                "capacity": 256,
                "read": 7,
                "write": 7,
            },
            {
                "n": 7,
                "steps": 2,
                "capacity": 128,
                "read": 20,
                "write": 21,
                **({"read_col": -1, "write_col": 2, "write_row": -24, "write": 80} if descriptor else {}),
            },
            {
                "n": 0,
                "capacity": 0,
                "read": -8192 if descriptor else -512,
                "write": 8192 if descriptor else 512,
            },
            {
                "n": 7,
                "steps": 0,
                "capacity": 17,
                "read": 8192 if descriptor else 512,
                "write": -8192 if descriptor else -512,
            },
        )
    ):
        values, expected = arguments(dtype, order, descriptor=descriptor, **options)
        replay(tmp_path / f"replay-{index}", qualified, values, expected)
    manifest = json.loads((qualified / "compiler/compile-inputs.json").read_text())
    assert manifest["payload_profile"] == "shared-source-word-buffer"


@pytest.mark.parametrize(
    "width,dtype,order",
    (
        (16, "f64", "little"),
        (16, "i64", "big"),
        (64, "f16", "little"),
        (64, "i16", "big"),
        (64, "f32", "little"),
        (64, "i32", "big"),
    ),
)
def test_source_word_width_is_independent_of_root_width(tmp_path, width, dtype, order):
    from test_mixed_effects import qualify

    from t2m.allocation_word_demo import convert
    from t2m.examples.descriptor_reference import policy

    result = convert(
        tmp_path / "source", width=width, root_dtype=dtype, byte_order=order
    )
    values, expected = arguments(dtype, order, width=width)
    qualify(result, tmp_path, values, policy(), [expected])
