"""Different-width byte views keep whole-word values and byte conflicts."""

import os

import numpy as np
import pytest
from test_descriptor_state import REQUIRED

from t2m.byte_buffer_demo import convert
from t2m.examples.byte_buffer_reference import arguments, policy

pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing native/shared/Merlin environments required"
)


@pytest.mark.parametrize(
    "dtype,order,scalar",
    [
        *(
            (dtype, order, False)
            for dtype in ("i16", "i32", "i64", "f16", "bf16", "f32", "f64")
            for order in ("little", "big")
        ),
        ("f64", "little", True),
        ("bf16", "big", True),
    ],
)
def test_native_and_merlin_byte_word_updates(tmp_path, dtype, order, scalar):
    from test_mixed_effects import qualify

    result = convert(tmp_path / "source", dtype=dtype, byte_order=order, scalar=scalar)
    width = result.spec.buffers[1].type.element_type.bitwidth
    block = 1 if scalar else 4
    values, expected = arguments(width, order, block=block)
    _, qualified = qualify(result, tmp_path, values, policy(), expected)
    assert "prov.logical_byte_views" in result.mlir_text
    reports = [
        r
        for r in result.trace["transformations"]
        if r["kind"] == "logical-byte-buffer-word-event"
    ]
    assert reports and all(
        r["byte_order"] == order and r["byte_tile"][-1] == width // 8 for r in reports
    )
    for index, options in enumerate(
        (
            {"n": 13, "steps": 3, "capacity": 512},
            {
                "n": 13,
                "steps": 3,
                "capacity": 512,
                "origin": 350,
                "stride": -24,
                "advance": -1,
            },
            {
                "n": 1,
                "steps": 3,
                "origin": 7,
                "stride": -128,
                "advance": -1,
                "capacity": 32,
            },
            {"n": 0, "capacity": 0, "origin": 8191},
            {"n": 0, "capacity": 17, "origin": -8192},
            {"n": 13, "steps": 0, "capacity": 17, "origin": 8191},
        )
    ):
        # The one-lane reversed case needs room before its signed recurrence.
        if index == 2:
            options["origin"] = 24
        values, expected = arguments(width, order, block=block, **options)
        replay(tmp_path / f"replay-{index}", qualified, values, expected)
    failures = [
        (
            {"n": 1, "steps": 1, "origin": 9, "capacity": 9 + width // 8 - 1},
            "Triton active memory index outside logical allocation",
        ),
        (
            {"n": 4, "steps": 1, "stride": 1},
            "Triton memory conflicts across logical programs"
            if scalar
            else "Triton scatter has conflicting active lanes",
        ),
        (
            {"n": 7, "steps": 2, "stride": width // 8, "advance": 4},
            "Triton memory conflicts across logical programs",
        ),
    ]
    for index, (options, message) in enumerate(failures):
        values, expected = arguments(width, order, block=block, invalid=True, **options)
        replay(
            tmp_path / f"refused-{index}", qualified, values, expected, message=message
        )


def replay(root, qualified, values, expected, *, message=None):
    import json
    import signal

    from t2m.invocation import invoke_kernel, verify_invocation

    root.mkdir()
    np.savez(root / "inputs.npz", **{f"in{i}": value for i, value in enumerate(values)})
    np.savez(
        root / "reference.npz", **{f"out{i}": value for i, value in enumerate(expected)}
    )
    reference = policy()
    reference["results"]["comparisons"] = [{"kind": "words"} for _ in expected]
    (root / "reference.json").write_text(json.dumps(reference))
    options = {
        "merlin_root": os.environ["T2M_MERLIN_ROOT"],
        "python": os.environ["T2M_MERLIN_PYTHON"],
    }
    if message is None:
        report = invoke_kernel(
            qualified,
            root / "inputs.npz",
            root / "reference.json",
            root / "invocation",
            **options,
        )
        verify_invocation(root / "invocation")
        assert report["native_compilation"] == "reused-qualified-library"
        with np.load(
            root / "invocation/evaluation/outputs.npz", allow_pickle=False
        ) as actual:
            for i, value in enumerate(expected):
                assert actual[f"out{i}"].tobytes() == value.tobytes()
    else:
        with pytest.raises(ValueError, match="actual effect outcome differs"):
            invoke_kernel(
                qualified,
                root / "inputs.npz",
                root / "reference.json",
                root / "invocation",
                **options,
            )
        outcome = root / "invocation/evaluation"
        assert (
            json.loads((outcome / "outcome.json").read_text())["returncode"]
            == -signal.SIGABRT
        )
        assert (outcome / "stdout.bin").read_bytes() == (message + "\n").encode()
        assert not (outcome / "outputs.npz").exists()


@pytest.mark.parametrize("dtype", ("f16", "bf16", "f32", "f64"))
@pytest.mark.parametrize("order", ("little", "big"))
def test_nonzero_and_signaling_fill_words_survive_byte_plane_masks(
    tmp_path, dtype, order
):
    from test_indexed_descriptors import replay as replay_one
    from test_mixed_effects import qualify

    from t2m.byte_buffer_demo import convert_fills
    from t2m.examples.byte_buffer_reference import fill_arguments
    from t2m.examples.descriptor_reference import policy as one_policy

    result = convert_fills(tmp_path / "source", dtype=dtype, byte_order=order)
    width = result.spec.buffers[1].type.element_type.bitwidth
    nan = {"f16": 0x7C11, "bf16": 0x7F81, "f32": 0x7F800011, "f64": 0x7FF0000000000011}[
        dtype
    ]
    values, expected = fill_arguments(width, order, fill=nan)
    _, qualified = qualify(result, tmp_path, values, one_policy(), [expected])
    for index, options in enumerate(
        (
            {"n": 13, "capacity": 512, "fill": nan | (1 << (width - 1))},
            {
                "n": 1,
                "capacity": width // 8,
                "origin": 0,
                "stride": 128,
                "fill": 1 << (width - 1),
            },
            {"n": 0, "capacity": 0, "origin": 8191, "fill": nan},
        )
    ):
        values, expected = fill_arguments(width, order, **options)
        replay_one(tmp_path / f"replay-{index}", qualified, values, expected)


@pytest.mark.parametrize("order", ("little", "big"))
def test_public_byte_demo_and_generic_source_cli_reach_merlin(tmp_path, order):
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification

    capture, qualified = tmp_path / "capture", tmp_path / "qualified"
    kernel = "byte-buffer" if order == "little" else "byte-buffer-big"
    assert (
        main(
            [
                "shared-demo",
                "--kernel",
                kernel,
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
    tensor_capture = tmp_path / "tensor-capture"
    assert (
        main(
            [
                "shared-tensor",
                str(capture / "source.ttir"),
                "--abi",
                str(capture / "spec.json"),
                "--byte-order",
                order,
                "--memory-profile",
                "guarded-buffer",
                "--address-policy",
                "source-words",
                "--optimizer",
                os.environ["TRITON_SHARED_OPT"],
                "--python",
                os.environ["T2M_NATIVE_PYTHON"],
                "--inputs",
                str(capture / "inputs.npz"),
                "--reference",
                str(capture / "reference.npz"),
                "--out",
                str(tensor_capture),
            ]
        )
        == 0
    )
    verify_bundle(tensor_capture)


def test_selected_byte_roots_keep_while_offsets_and_both_states(tmp_path):
    from test_mixed_effects import qualify

    from t2m.byte_buffer_demo import convert_selected
    from t2m.examples.byte_buffer_reference import selected_arguments

    result = convert_selected(tmp_path / "source")
    reference = policy()
    reference["results"]["comparisons"] *= 1
    reference["results"]["comparisons"].append({"kind": "words"})
    values, expected = selected_arguments()
    _, qualified = qualify(result, tmp_path, values, reference, expected)
    assert "scf.while" in result.mlir_text
    assert any(
        r.get("kind") == "logical-byte-buffer-word-event" and r["roots"] == [0, 1]
        for r in result.trace["transformations"]
    )
    for index, options in enumerate(
        (
            {"n": 13, "steps": 3, "capacity": 512, "choice": False},
            {
                "n": 13,
                "steps": 3,
                "capacity": 512,
                "origin": 350,
                "stride": -24,
                "advance": -1,
            },
            {"n": 0, "capacity": 0, "origin": 8191},
            {"steps": 0, "capacity": 17, "origin": 8191},
        )
    ):
        values, expected = selected_arguments(**options)
        replay(tmp_path / f"replay-{index}", qualified, values, expected)


@pytest.mark.parametrize("descriptor", (False, True))
def test_original_typed_parameters_bind_one_byte_root(tmp_path, descriptor):
    from test_indexed_descriptors import replay as replay_one
    from test_mixed_effects import qualify

    from t2m.examples.byte_buffer_reference import bound_arguments
    from t2m.examples.descriptor_reference import policy as one_policy

    if descriptor:
        from t2m.descriptor_state_demo import convert

        result = convert(
            tmp_path / "source",
            dtype="i32",
            word_views=True,
            byte_storage=True,
            byte_order="big",
        )
    else:
        from t2m.allocation_word_demo import convert

        result = convert(tmp_path / "source", byte_storage=True, byte_order="big")
    assert result.spec.buffers[0].dtype == "i8"
    assert any(
        b.get("representation") == "byte-buffer-view-v1"
        for b in result.trace["source_argument_binding"]["bindings"]
    )
    values, expected = bound_arguments(descriptor=descriptor)
    _, qualified = qualify(result, tmp_path, values, one_policy(), [expected])
    for index, options in enumerate(
        (
            {
                "n": 13 if not descriptor else 19,
                "m": 5,
                "capacity": 256 if not descriptor else 768,
                "write": 8,
                "steps": 3,
            },
            {"n": 1, "m": 1, "read": 23, "write": 7},
            {"n": 0, "m": 0, "capacity": 0, "read": 511, "write": -512},
            {"steps": 0, "read": 511, "write": -512},
        )
    ):
        values, expected = bound_arguments(descriptor=descriptor, **options)
        replay_one(tmp_path / f"replay-{index}", qualified, values, expected)
