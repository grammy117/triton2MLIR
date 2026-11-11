"""Different-width parameter bindings retain actual byte bounds and conflicts."""

import pytest
from test_descriptor_state import REQUIRED

from t2m.examples.typed_parameter_reference import arguments

pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


@pytest.mark.parametrize("descriptor", (False, True))
def test_typed_root_parameter_guards_refuse_invalid_source_effects(
    tmp_path, descriptor
):
    from test_indexed_descriptors import replay
    from test_mixed_effects import qualify

    from t2m.examples.descriptor_reference import policy

    dtype, order = "f16", "big"
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
    values, expected = arguments(dtype, order, descriptor=descriptor)
    _, qualified = qualify(result, tmp_path, values, policy(), [expected])
    failures = [
        (
            {"n": 1, "m": 1, "steps": 1, "capacity": 1, "read": 0, "write": 0},
            "Triton active memory index outside logical allocation",
        ),
        (
            {
                "n": 19 if descriptor else 15,
                "steps": 2,
                "capacity": 128,
                "read": 9,
                "write": 10,
            },
            "Triton memory conflicts across logical programs",
        ),
    ]
    if descriptor:
        failures.append(
            ({"write_col": 0}, "Triton scatter has conflicting active lanes")
        )
    for index, (options, message) in enumerate(failures):
        values, expected = arguments(
            dtype, order, descriptor=descriptor, invalid=True, **options
        )
        replay(
            tmp_path / f"refused-{index}", qualified, values, expected, message=message
        )
