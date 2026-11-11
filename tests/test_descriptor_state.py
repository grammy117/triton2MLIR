"""Descriptor origins and shared state execute with complete portable source custody."""

import json
import os

import pytest

from t2m.descriptor_state_demo import convert
from t2m.examples.descriptor_reference import policy
from t2m.examples.descriptor_state_reference import arguments

REQUIRED = all(
    os.environ.get(name)
    for name in (
        "TRITON_SHARED_OPT",
        "T2M_NATIVE_PYTHON",
        "T2M_MERLIN_ROOT",
        "T2M_MERLIN_PYTHON",
        "T2M_MERLIN_CLANG",
    )
)
pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


@pytest.mark.parametrize("dtype", ("i64", "f32", "bf16"))
def test_shared_descriptor_origins_retain_tile_snapshots_and_full_state(
    tmp_path, dtype
):
    from test_indexed_descriptors import replay
    from test_mixed_effects import qualify

    from t2m.source_capsule import members

    result = convert(tmp_path / "source", dtype=dtype)
    binding = result.trace["source_argument_binding"]
    assert (
        binding["descriptor_bindings"]["schema"] == "t2m.source-allocation-arguments.v2"
    )
    assert len(binding["descriptor_bindings"]["arguments"]) == 2
    assert any(argument["kind"] == "sext" for argument in binding["bindings"])
    assert len(result.spec.buffers) == len(result.spec.outputs) == 1
    entry = next(iter(result.module.ops))
    assert (
        json.loads(entry.attributes["prov.kernel_descriptor_state_bindings"].data)[
            "grants_semantics"
        ]
        is False
    )
    values, expected = arguments(dtype)
    capture, qualified = qualify(result, tmp_path, values, policy(), [expected])
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    assert receipt["source_closure_verified"] is False
    for name in members(result.trace):
        assert name in receipt["artifacts"]
        assert (capture / name).read_bytes() == (
            qualified / "compiler/source" / name
        ).read_bytes()
    assert (capture / "allocation-original.ttir").read_bytes() == (
        tmp_path / "source/source.ttir"
    ).read_bytes()
    assert "!tt.tensordesc" in (capture / "allocation-original.ttir").read_text()
    assert "!tt.tensordesc" not in (capture / "source.ttir").read_text()
    cases = [
        {"m": 5, "n": 19, "steps": 3, "capacity": 256, "write_origin": 8},
        {
            "m": 5,
            "n": 19,
            "steps": 2,
            "capacity": 256,
            "read_origin": 144,
            "write_origin": 144,
            "read_strides": (-24, -1),
            "write_strides": (-24, -1),
        },
        {"steps": 0},
        {"m": 0, "capacity": 0, "read_origin": 8191, "write_origin": 8191},
        {"m": 0, "capacity": 17, "read_origin": 8191, "write_origin": 8191},
        {"n": 0, "capacity": 17},
        {"m": 1, "n": 1, "read_origin": 23, "write_origin": 7},
        {"m": 2, "n": 13, "steps": 3, "read_origin": 16, "write_origin": 8},
    ]
    for index, options in enumerate(cases):
        values, expected = arguments(dtype, **options)
        replay(tmp_path / f"replay-{index}", qualified, values, expected)
    failures = [
        (
            {"read_origin": 8191},
            "Triton active memory index outside logical allocation",
        ),
        (
            {"write_origin": 8191},
            "Triton active memory index outside logical allocation",
        ),
        ({"write_strides": (24, 0)}, "Triton scatter has conflicting active lanes"),
        (
            {"m": 5, "n": 19, "capacity": 256},
            "Triton memory conflicts across logical programs",
        ),
    ]
    for index, (options, message) in enumerate(failures):
        values, expected = arguments(dtype, invalid=True, **options)
        replay(
            tmp_path / f"refused-{index}", qualified, values, expected, message=message
        )
