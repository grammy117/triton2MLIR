"""Literal words compose with runtime tiled state through the real consumer."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.examples.literal_state import literal_words, source
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
from t2m.tensor_storage import DTYPES

REQUIRED = all(
    os.environ.get(key)
    for key in (
        "TRITON_SHARED_OPT",
        "T2M_NATIVE_PYTHON",
        "T2M_MERLIN_ROOT",
        "T2M_MERLIN_PYTHON",
        "T2M_MERLIN_CLANG",
    )
)
pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin tools required"
)


def capture(root, dtype, profile="guarded-buffer", allocated=False):
    path = root / "source.ttir"
    path.write_text(source(dtype))
    abi = TensorABI(
        "literal_state",
        (
            TensorBuffer("x", ("n",), dtype=dtype),
            TensorBuffer("state", ("n",), "readwrite", dtype),
            TensorBuffer("snapshot", ("n",), "write", dtype),
        ),
        grid=(GridExtent("n", 8),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 3, bounds=(0, 31)),
            RuntimeScalar("steps", "i32", 4, bounds=(0, 3)),
        ),
    )
    options = {
        "optimizer": os.environ["TRITON_SHARED_OPT"],
        "memory_profile": profile,
        "address_policy": "source-words" if profile == "guarded-buffer" else "structured",
    }
    if allocated:
        from t2m.allocation_bindings import (
            PointerArgument,
            ScalarArgument,
            convert_allocations,
        )

        return convert_allocations(
            path,
            replace(abi, entry="allocation_state"),
            (
                PointerArgument("x"),
                PointerArgument("state"),
                PointerArgument("snapshot"),
                ScalarArgument("n"),
                ScalarArgument("steps"),
            ),
            root / "translation",
            source_entry="literal_state",
            **options,
        )
    return convert_shared(path, abi, root / "translation", **options)


def arguments(dtype, n=19, steps=3):
    bits, sentinel, table = literal_words(dtype)
    word_type = np.dtype(f"uint{bits}")
    x_words = np.resize(np.asarray(table[::-1], dtype=word_type), n)
    state_words = np.resize(
        np.asarray([1, 0, 1 << (bits - 1), sentinel], dtype=word_type), n
    )
    updated, saved = state_words.copy(), state_words.copy()
    for index in range(n):
        if index % 2 == 0:
            saved[index] = sentinel
        for step in range(steps):
            if (step + index // 8) % 2 == 0:
                if index % 2 == 0:
                    updated[index] = table[index % 8]
            elif index % 2:
                updated[index] = x_words[index]

    def array(words):
        return words.copy() if dtype == "bf16" else words.view(DTYPES[dtype]).copy()

    return [
        array(x_words),
        array(state_words),
        np.asarray(n, np.int32),
        np.asarray(steps, np.int32),
    ], [array(updated), array(saved)]


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_tiled_literal_state_retains_words_and_reuses_runtime_library(
    tmp_path, dtype, profile
):
    from test_mixed_effects import expected, qualify

    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    result = capture(tmp_path, dtype, profile)
    assert "scf.for" in result.mlir_text and "scf.if" in result.mlir_text
    assert (
        len(
            [
                report
                for report in result.trace["transformations"]
                if report["kind"] == "source-floating-literal-words"
            ]
        )
        == 2
    )
    inputs, outputs = arguments(dtype)
    policy = expected("", [{"kind": "words"}] * 2)
    captured, compiled = qualify(result, tmp_path, inputs, policy, outputs)
    library = _record(compiled / "host/model.so")
    for n, steps in [(0, 3), (1, 0), (31, 1), (13, 2)]:
        caller = tmp_path / f"caller-{n}"
        caller.mkdir()
        inputs, outputs = arguments(dtype, n, steps)
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
    from scripts.qualify_spike_effect import qualify as target
    from scripts.qualify_spike_effect import verify

    options = {
        "merlin_root": os.environ["T2M_MERLIN_ROOT"],
        "python": os.environ["T2M_MERLIN_PYTHON"],
    }
    target(captured, tmp_path / "target", **options)
    verify(tmp_path / "target", **options)


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
def test_allocation_source_protects_literals_before_wrapper_normalization(
    tmp_path, dtype
):
    from test_mixed_effects import expected, qualify

    result = capture(tmp_path, dtype, allocated=True)
    protection = json.loads(result.auxiliary_artifacts["allocation-protection.json"])
    assert len(protection["literal_conversions"]) == 2
    assert (
        result.trace["source_argument_binding"]["source_capsule"]["schema"]
        == "t2m.source-allocation-capsule.v2"
    )
    inputs, outputs = arguments(dtype)
    captured, _ = qualify(
        result, tmp_path, inputs, expected("", [{"kind": "words"}] * 2), outputs
    )
    from scripts.qualify_spike_effect import qualify as target
    from scripts.qualify_spike_effect import verify

    options = {
        "merlin_root": os.environ["T2M_MERLIN_ROOT"],
        "python": os.environ["T2M_MERLIN_PYTHON"],
    }
    target(captured, tmp_path / "target", **options)
    verify(tmp_path / "target", **options)
