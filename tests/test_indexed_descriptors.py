"""Official descriptor rewrites retain indexed state through actual Merlin calls."""

import json
import os
import signal

import numpy as np
import pytest

from t2m.descriptor_demo import convert
from t2m.examples.descriptor_reference import (
    policy,
    reduction_arguments,
    transfer_arguments,
)

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = (
    OPT
    and NATIVE
    and all(
        os.environ.get(name)
        for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
    )
)
pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


def replay(root, qualified, values, expected, *, message=None):
    from t2m.invocation import invoke_kernel, verify_invocation

    root.mkdir()
    np.savez(root / "inputs.npz", **{f"in{i}": value for i, value in enumerate(values)})
    np.savez(root / "reference.npz", out0=expected)
    (root / "reference.json").write_text(json.dumps(policy()))
    kwargs = {
        "merlin_root": os.environ["T2M_MERLIN_ROOT"],
        "python": os.environ["T2M_MERLIN_PYTHON"],
    }
    if message is None:
        before = [value.tobytes() for value in values]
        report = invoke_kernel(
            qualified,
            root / "inputs.npz",
            root / "reference.json",
            root / "invocation",
            **kwargs,
        )
        verify_invocation(root / "invocation")
        assert report["native_compilation"] == "reused-qualified-library"
        with np.load(
            root / "invocation/evaluation/outputs.npz", allow_pickle=False
        ) as actual:
            assert actual.files == ["out0"]
            assert (
                actual["out0"].dtype == expected.dtype
                and actual["out0"].shape == expected.shape
            )
            assert actual["out0"].tobytes() == expected.tobytes()
        assert [value.tobytes() for value in values] == before
    else:
        with pytest.raises(ValueError, match="actual effect outcome differs"):
            invoke_kernel(
                qualified,
                root / "inputs.npz",
                root / "reference.json",
                root / "invocation",
                **kwargs,
            )
        evaluation = root / "invocation/evaluation"
        assert (
            json.loads((evaluation / "outcome.json").read_text())["returncode"]
            == -signal.SIGABRT
        )
        assert (evaluation / "stdout.bin").read_bytes() == (message + "\n").encode()
        assert (evaluation / "invocation-started.json").is_file()
        assert not (evaluation / "outputs.npz").exists()
        assert not (root / "invocation/invocation.json").exists()


TRANSFERS = [
    (kind, dtype, "zero")
    for kind in ("gather", "scatter")
    for dtype in ("i16", "i32", "f16", "bf16", "f32")
] + [("gather", dtype, "nan") for dtype in ("f16", "bf16", "f32")]


@pytest.mark.parametrize("kind,dtype,padding", TRANSFERS)
def test_indexed_transfer_preserves_padding_words_and_conflicts(
    tmp_path, kind, dtype, padding
):
    from test_mixed_effects import qualify

    result = convert(
        tmp_path / "source", kind=kind, dtype=dtype, padding=padding, optimizer=OPT
    )
    assert any(
        r["opcode"] == "tt.descriptor_" + kind
        for r in result.trace["source_operations"]
    )
    assert any(
        r["opcode"] == "tt.make_tensor_descriptor"
        for r in result.trace["source_operations"]
    )
    assert all(
        not op.name.startswith(("tt.", "tts.", "gpu.", "llvm."))
        for op in result.module.walk()
    )
    assert "triton-rewrite-tensor-descriptor-to-pointer" in json.dumps(
        result.trace["transformations"]
    )
    assert result.spec.intent in result.mlir_text
    contract = next(
        t for t in result.trace["transformations"] if "descriptor_profile" in t
    )
    assert contract["descriptor_profile"]["name"] == "logical-tensor-descriptors-v1"
    assert contract["descriptor_profile"]["grants_semantics"] is False
    assert (
        contract["descriptor_implementation"]["source_reference_is_build_receipt"]
        is False
    )
    assert len(contract["descriptor_implementation"]["binary_sha256"]) == 64
    entry = next(iter(result.module.ops))
    intent = json.loads(entry.attributes["prov.kernel_original_descriptors"].data)
    assert intent["grants_semantics"] is False
    assert any(r["opcode"] == "tt.descriptor_" + kind for r in intent["operations"])
    values, expected = transfer_arguments(kind, dtype, padding=padding)
    _, qualified = qualify(result, tmp_path, values, policy(), [expected])
    cases = [
        {"count": 0},
        {"rows": 0, "count": 3},
        {"cols": 0, "count": 3},
        {"rows": 3, "cols": 11, "count": 5, "y": -4, "stride": 16},
        {"rows": 9, "cols": 19, "count": 17, "y": 16, "stride": 32},
        {"count": 3, "y": 32},
    ]
    for index, changes in enumerate(cases):
        values, expected = transfer_arguments(kind, dtype, padding=padding, **changes)
        replay(tmp_path / f"replay-{index}", qualified, values, expected)
    if kind == "scatter":
        for index, selected in enumerate(([1, 1], [1, 2, 3, 4, 5, 6, 0, -1, 1])):
            values, expected = transfer_arguments(
                kind, dtype, count=len(selected), selected=selected
            )
            replay(
                tmp_path / f"conflict-{index}",
                qualified,
                values,
                expected,
                message=(
                    "Triton scatter has conflicting active lanes"
                    if index == 0
                    else "Triton memory conflicts across logical programs"
                ),
            )


REDUCTIONS = (
    [(kind, "i32", False) for kind in ("add", "min", "max", "and", "or", "xor")]
    + [(kind, "i64", False) for kind in ("min", "max")]
    + [(kind, "i32", True) for kind in ("min", "max")]
    + [("add", "f32", False)]
)


@pytest.mark.parametrize("kind,dtype,unsigned", REDUCTIONS)
def test_reducing_descriptors_retain_collisions_recurrence_and_initial_state(
    tmp_path, kind, dtype, unsigned
):
    from test_mixed_effects import qualify

    result = convert(
        tmp_path / "source",
        kind=kind,
        dtype=dtype,
        unsigned=unsigned,
        optimizer=OPT,
        f32_atomic_add="ieee" if dtype == "f32" else None,
    )
    assert any(
        r["opcode"] == "tt.descriptor_reduce" for r in result.trace["source_operations"]
    )
    assert any(
        r["opcode"] == "tt.atomic_rmw" for r in result.trace["normalized_operations"]
    )
    assert "scf.for" in result.mlir_text
    assert not any(
        op.name.startswith(("tt.", "tts.", "gpu.", "llvm."))
        for op in result.module.walk()
    )
    values, expected = reduction_arguments(kind, dtype, unsigned=unsigned)
    _, qualified = qualify(result, tmp_path, values, policy(), [expected])
    cases = [
        {"tiles": 0},
        {"steps": 0},
        {"rows": 0},
        {"cols": 0},
        {"rows": 9, "tiles": 2, "steps": 3, "row": 8, "y": -4, "stride": 32},
        {"rows": 3, "tiles": 1, "steps": 1, "row": 0, "y": 16, "stride": 24},
    ]
    for index, changes in enumerate(cases):
        values, expected = reduction_arguments(
            kind, dtype, unsigned=unsigned, **changes
        )
        replay(tmp_path / f"replay-{index}", qualified, values, expected)


@pytest.mark.parametrize("kind", ("gather", "scatter", "reduce"))
def test_descriptor_cli_packages_reach_the_merlin_interface(tmp_path, kind):
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification

    capture, qualified = tmp_path / "capture", tmp_path / "qualified"
    assert (
        main(
            [
                "shared-demo",
                "--kernel",
                "descriptor-" + kind,
                "--memory-profile",
                "guarded-buffer",
                "--address-policy",
                "source-words",
                "--out",
                str(capture),
                "--python",
                NATIVE,
            ]
        )
        == 0
    )
    verify_bundle(capture)
    differences = json.loads((capture / "compatibility.json").read_text())[
        "differences"
    ]
    assert any("logical base/shape/element-strides" in entry for entry in differences)
    assert "prov.kernel_original_descriptors" in (capture / "model.mlir").read_text()
    qualify_kernel(
        capture,
        qualified,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,
        clang=os.environ["T2M_MERLIN_CLANG"],
    )
    verify_kernel_qualification(qualified)
