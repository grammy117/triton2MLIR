"""General original TTIR reductions reach all memory profiles and Merlin."""

import json
import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.examples.reduction_reference import recurrence
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

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
    not REQUIRED, reason="existing native/shared/Merlin tools required"
)


def capture(root, kind, axis=1, profile="guarded-buffer"):
    from t2m.examples.reduction_kernels import reduce_recurrence
    from t2m.frontend import capture_ttir

    types = ("f32", "i32") if kind == "argmin" else ("i64", "i64")
    pointer = {"f32": "*fp32", "i32": "*i32", "i64": "*i64"}
    source = root / "source.ttir"
    source.write_text(
        capture_ttir(
            reduce_recurrence,
            {
                "x": pointer[types[0]],
                "y": pointer[types[1]],
                "out_a": pointer[types[0]],
                "out_b": pointer[types[1]],
                "n": "i32",
                "steps": "i32",
                "B": "constexpr",
                "KIND": "constexpr",
                "AXIS": "constexpr",
            },
            {"B": 8, "KIND": kind, "AXIS": axis},
        )
    )
    abi = TensorABI(
        "reduce_recurrence",
        tuple(
            TensorBuffer(
                name, ("n",), "read" if index < 2 else "write", types[index % 2]
            )
            for index, name in enumerate(("x", "y", "out_a", "out_b"))
        ),
        grid=(GridExtent("n", 8),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
            RuntimeScalar("steps", "i32", 5, bounds=(0, 3)),
        ),
    )
    return convert_shared(
        source,
        abi,
        root / "translation",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )


def arguments(kind, axis=1, n=19, steps=2):
    if kind == "argmin":
        x = np.resize(
            np.asarray(
                [
                    0x80000000,
                    0x7FC00001,
                    0x3F800000,
                    0xBF800000,
                    0xFFC00013,
                    0xFF800000,
                    0,
                    0xBF800000,
                ],
                np.uint32,
            ),
            n,
        ).view(np.float32)
        y = np.resize(np.asarray([13, -7, 5, 9, 2, -11, 7, 4], np.int32), n)
    else:
        x = np.resize(
            np.asarray([3, -1, (1 << 62) + 1, 7, -17, 21, 35, 0], np.int64), n
        )
        y = np.resize(
            np.asarray([(1 << 62) + 3, 7, -19, 13, -1, 0, 35, 21], np.int64), n
        )
    return [x, y, np.asarray(n, np.int32), np.asarray(steps, np.int32)], recurrence(
        x, y, n, steps, kind, axis
    )


@pytest.mark.parametrize("kind", ["and", "gcd", "affine", "argmin"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_original_combiner_and_state_reach_merlin_and_target(
    tmp_path, kind, axis, profile
):
    from test_mixed_effects import expected, qualify

    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    result = capture(tmp_path, kind, axis, profile)
    assert any(
        report["kind"] == "ttir-first-element-reduction"
        for report in result.trace["transformations"]
    )
    assert (
        "scf.for" in result.mlir_text and "first-original-element" in result.mlir_text
    )
    if kind == "gcd":
        assert "scf.while" in result.mlir_text
    if kind == "argmin":
        assert "scf.if" in result.mlir_text
    inputs, outputs = arguments(kind, axis)
    policy = expected("", [{"kind": "words"}] * 2)
    captured, compiled = qualify(result, tmp_path, inputs, policy, outputs)
    library = _record(compiled / "host/model.so")
    for n, steps in [(0, 3), (1, 0), (31, 1), (13, 3)]:
        caller = tmp_path / f"caller-{n}"
        caller.mkdir()
        inputs, outputs = arguments(kind, axis, n, steps)
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(inputs)}
        )
        np.savez(
            caller / "reference.npz",
            **{f"out{i}": value for i, value in enumerate(outputs)},
        )
        (caller / "reference.json").write_text(json.dumps(policy))
        out = tmp_path / f"invoke-{n}"
        invoke_kernel(
            compiled,
            caller / "inputs.npz",
            caller / "reference.json",
            out,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(out)
        assert _record(out / "source/host/model.so") == library
    from scripts import qualify_spike_effect as target

    options = {
        "merlin_root": os.environ["T2M_MERLIN_ROOT"],
        "python": os.environ["T2M_MERLIN_PYTHON"],
    }
    target.qualify(captured, tmp_path / "target", **options)
    target.verify(tmp_path / "target", **options)
