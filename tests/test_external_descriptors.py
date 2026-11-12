"""Descriptor arguments are actual logical fields at the Merlin compiler boundary."""

import json
import os

import pytest

from t2m.examples.descriptor_reference import policy
from t2m.examples.external_descriptor_reference import arguments
from t2m.external_descriptor_demo import convert

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

CASES = [
    ("gather", "f32", (2, 16)),
    ("gather", "bf16", (2, 16)),
    ("scatter", "f32", (2, 16)),
    ("scatter", "bf16", (2, 16)),
    *((kind, "i32", (2, 16)) for kind in ("add", "min", "max", "and", "or", "xor")),
    ("min", "i64", (2, 16)),
    ("add", "f32", (2, 16)),
    ("load", "i64", (16,)),
    ("store", "i64", (16,)),
    ("load", "f32", (16,)),
    ("store", "f32", (16,)),
    ("load", "bf16", (2, 2, 16)),
    ("store", "bf16", (2, 2, 16)),
]


@pytest.mark.parametrize("kind,dtype,shape", CASES)
def test_external_fields_keep_padding_strides_state_and_runtime_flags(
    tmp_path, kind, dtype, shape
):
    from test_indexed_descriptors import replay
    from test_mixed_effects import qualify

    result = convert(tmp_path / "source", kind=kind, dtype=dtype, shape=shape)
    binding = result.trace["descriptor_argument_binding"]
    assert (
        binding["schema"] == "t2m.logical-descriptor-arguments.v1"
        and binding["grants_semantics"] is False
    )
    entry = next(iter(result.module.ops))
    assert (
        json.loads(entry.attributes["prov.kernel_descriptor_arguments"].data)["entry"]
        == result.spec.entry
    )
    descriptor, *ordinary = binding["arguments"]
    assert descriptor["source_argument"] == 0 and descriptor["descriptor"] is True
    rank = len(shape) if kind in ("load", "store") else 2
    assert len(descriptor["fields"]) == 3 + 2 * rank
    assert (
        len(ordinary) >= 2 * rank
    )  # Real AST shape/stride metadata remains ordinary source input.
    assert not any(
        op.name.startswith(("tt.", "tts.", "gpu.", "llvm."))
        for op in result.module.walk()
    )
    defaults = (
        {"dimensions": (19,), "strides": (1,)}
        if rank == 1
        else {
            "dimensions": (3, 3, 19),
            "strides": (96, 24, 1),
            "offsets": (0, 0, 4),
            "capacity": 512,
        }
        if rank == 3
        else {}
    )
    values, expected = arguments(kind, dtype, block=shape, **defaults)
    _, qualified = qualify(result, tmp_path, values, policy(), [expected])
    if rank == 2:
        changes = [
            {"padding_nan": True, "round_tf32": True},
            {"dimensions": (0, 19), "capacity": 0},
            {"dimensions": (3, 11), "strides": (1, 9), "offsets": (-1, -4, 0)},
            {"dimensions": (9, 19), "strides": (24, 1), "offsets": (0, 16, 0)},
            {"capacity": 512, "count": 17, "tiles": 2, "steps": 3},
        ]
        changes += [{"count": 0}] if kind in ("gather", "scatter") else [{"steps": 0}]
    elif rank == 1:
        changes = [
            {"padding_nan": True, "offsets": (-4, 0, 0)},
            {"dimensions": (0,), "capacity": 0},
            {
                "dimensions": (19,),
                "strides": (3,),
                "offsets": (16, 0, 0),
                "capacity": 64,
            },
        ]
    else:
        changes = [
            {"padding_nan": True, "offsets": (-1, 0, 16)},
            {"dimensions": (0, 3, 19), "capacity": 0},
            {"dimensions": (3, 3, 19), "strides": (1, 3, 9), "offsets": (2, 2, 16)},
        ]
    for index, change in enumerate(changes):
        values, expected = arguments(kind, dtype, block=shape, **(defaults | change))
        replay(tmp_path / f"replay-{index}", qualified, values, expected)
    if kind == "scatter":
        values, expected = arguments(kind, dtype, count=2, selected=[1, 1])
        replay(
            tmp_path / "conflict",
            qualified,
            values,
            expected,
            message="Triton scatter has conflicting active lanes",
        )
    values, expected = arguments(
        kind, dtype, block=shape, **(defaults | {"capacity": 1, "invalid": True})
    )
    replay(
        tmp_path / "bounds",
        qualified,
        values,
        expected,
        message="Triton active memory index outside logical allocation",
    )


@pytest.mark.parametrize("kind", ("gather", "scatter", "reduce", "load", "store", "matmul"))
def test_external_descriptor_cli_uses_a_real_source_signature(tmp_path, kind):
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification

    capture, qualified = tmp_path / "capture", tmp_path / "qualified"
    assert (
        main(
            [
                "shared-demo",
                "--kernel",
                "external-descriptor-" + kind,
                "--memory-profile",
                "guarded-buffer",
                "--address-policy",
                "source-words",
                "--out",
                str(capture),
                "--python",
                os.environ["T2M_NATIVE_PYTHON"],
            ]
        )
        == 0
    )
    verify_bundle(capture)
    trace = json.loads((capture / "provenance.json").read_text())
    assert (
        trace["descriptor_argument_binding"]["schema"]
        == "t2m.logical-descriptor-arguments.v1"
    )
    assert "!tt.tensordesc" in (capture / "source.ttir").read_text()
    differences = json.loads((capture / "compatibility.json").read_text())[
        "differences"
    ]
    assert any("one-to-many logical field ABI" in item for item in differences)
    qualify_kernel(
        capture,
        qualified,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"],
    )
    verify_kernel_qualification(qualified)


@pytest.mark.parametrize("dtype", ("f32", "bf16"))
def test_two_external_descriptors_retain_tiled_k_reuse_and_original_c(tmp_path, dtype):
    from test_indexed_descriptors import replay
    from test_mixed_effects import qualify

    from t2m.examples.external_descriptor_reference import matmul_arguments
    from t2m.external_descriptor_demo import convert_matmul

    result = convert_matmul(tmp_path / "source", dtype=dtype)
    descriptors = [
        argument
        for argument in result.trace["descriptor_argument_binding"]["arguments"]
        if argument["descriptor"]
    ]
    assert [argument["block_shape"] for argument in descriptors] == [[16, 32], [32, 16]]
    contractions = [op for op in result.module.walk() if op.name == "linalg.matmul"]
    assert len(contractions) == 1
    parent = contractions[0].parent_op()
    while parent is not None and parent.name != "scf.for":
        parent = parent.parent_op()
    assert parent is not None
    values, expected = matmul_arguments(17, 19, 37, dtype=dtype)
    _, qualified = qualify(result, tmp_path, values, policy(), [expected])
    cases = [
        (0, 19, 0, None, False),
        (17, 0, 0, None, False),
        (1, 1, 0, None, False),
        (17, 19, 0, None, False),
        (17, 19, 37, (1, 20, 1, 42), False),
        (35, 21, 65, None, False),
        (3, 4, 33, (0, 2, 7, 1), False),
        (17, 19, 37, None, True),
    ]
    for index, (m, n, k, strides, sensitive) in enumerate(cases):
        values, expected = matmul_arguments(
            m, n, k, dtype=dtype, strides=strides, sensitive=sensitive
        )
        replay(tmp_path / f"replay-{index}", qualified, values, expected)
    values, expected = matmul_arguments(17, 19, 37, dtype=dtype)
    values[0] = values[0][:1].copy()
    values[-2] = values[-2] * 0 + 1
    replay(
        tmp_path / "bounds",
        qualified,
        values,
        expected,
        message="Triton active memory index outside logical allocation",
    )
