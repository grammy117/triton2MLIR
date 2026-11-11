"""Aliased typed arguments preserve one memory version through actual Merlin."""

import pytest
from test_descriptor_state import REQUIRED

from t2m.allocation_word_demo import convert
from t2m.examples.allocation_word_reference import arguments
from t2m.examples.descriptor_reference import policy

pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


@pytest.mark.parametrize(
    "width,storage,bfloat",
    (
        (16, "integer", False),
        (32, "integer", False),
        (64, "integer", False),
        (16, "float", False),
        (32, "float", False),
        (64, "float", False),
        (16, "integer", True),
        (16, "float", True),
    ),
)
def test_mixed_typed_source_arguments_observe_shared_word_updates(
    tmp_path, width, storage, bfloat
):
    from test_indexed_descriptors import replay
    from test_mixed_effects import qualify

    result = convert(
        tmp_path / "source",
        width=width,
        storage=storage,
        bfloat=bfloat,
        while_=storage == "integer",
    )
    views = [
        record
        for record in result.trace["source_argument_binding"]["bindings"]
        if record.get("representation") == "equal-width-word-view-v1"
    ]
    assert len(views) == 1
    assert "prov.logical_word_views" in result.mlir_text
    assert (
        "scf.while" in result.mlir_text
        if storage == "integer"
        else "scf.for" in result.mlir_text
    )
    # Partial sharing must stay within one program; a multi-program race refuses.
    values, expected = arguments(width, storage, bfloat, n=7)
    capture, qualified = qualify(result, tmp_path, values, policy(), [expected])
    for index, options in enumerate(
        (
            {"n": 31, "steps": 4, "write": 8},
            {"n": 7, "steps": 3, "read": 10, "write": 8},
            {"n": 1, "steps": 3, "read": 23, "write": 7},
            {"n": 0, "capacity": 0, "read": 511, "write": -512},
            {"n": 0, "capacity": 17, "read": 511, "write": -512},
            {"steps": 0, "read": 511, "write": -512},
        )
    ):
        values, expected = arguments(width, storage, bfloat, **options)
        replay(tmp_path / f"replay-{index}", qualified, values, expected)
    for index, (options, message) in enumerate(
        (
            (
                {"n": 1, "read": 511},
                "Triton active memory index outside logical allocation",
            ),
            (
                {"n": 1, "write": -512},
                "Triton active memory index outside logical allocation",
            ),
            ({"n": 13}, "Triton memory conflicts across logical programs"),
        )
    ):
        values, expected = arguments(width, storage, bfloat, invalid=True, **options)
        replay(
            tmp_path / f"refused-{index}", qualified, values, expected, message=message
        )
    from t2m.bundle import verify_bundle

    verify_bundle(capture)


def test_differently_typed_external_descriptors_share_one_word_root(tmp_path):
    from test_indexed_descriptors import replay
    from test_mixed_effects import qualify

    from t2m.descriptor_state_demo import convert
    from t2m.examples.allocation_word_reference import descriptor_arguments

    result = convert(tmp_path / "source", dtype="i32", word_views=True)
    binding = result.trace["source_argument_binding"]
    assert len(binding["descriptor_bindings"]["arguments"]) == 2
    assert "prov.logical_word_views" in result.mlir_text
    values, expected = descriptor_arguments()
    _, qualified = qualify(result, tmp_path, values, policy(), [expected])
    for index, options in enumerate(
        (
            {"m": 5, "n": 19, "capacity": 256, "write": 8},
            {"read": 16, "write": 8, "steps": 3},
            {"m": 0, "capacity": 0, "read": 511, "write": -512},
            {"steps": 0, "read": 511, "write": -512},
        )
    ):
        values, expected = descriptor_arguments(**options)
        replay(tmp_path / f"replay-{index}", qualified, values, expected)


def test_allocation_word_public_demo_reaches_merlin(tmp_path):
    import os

    from t2m.bundle import verify_bundle
    from t2m.cli import main
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification

    capture, qualified = tmp_path / "capture", tmp_path / "qualified"
    assert (
        main(
            [
                "shared-demo",
                "--kernel",
                "allocation-word-state",
                "--out",
                str(capture),
                "--memory-profile",
                "guarded-buffer",
                "--address-policy",
                "source-words",
                "--optimizer",
                os.environ["TRITON_SHARED_OPT"],
                "--python",
                os.environ["T2M_NATIVE_PYTHON"],
            ]
        )
        == 0
    )
    verify_bundle(capture)
    qualify_kernel(
        capture,
        qualified,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"],
    )
    verify_kernel_qualification(qualified)
