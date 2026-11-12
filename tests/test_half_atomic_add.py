"""IEEE half atomics keep rounded updates and exact old storage words."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT = os.environ.get("TRITON_SHARED_OPT")
NATIVE = os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(
    os.environ.get(name)
    for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
)
pytestmark = pytest.mark.skipif(
    not (OPT and NATIVE), reason="existing shared/native tools required"
)


def capture(root, dtype="f16", scalar=False):
    from t2m.examples.half_atomic_kernels import half_atomic_add
    from t2m.frontend import capture_ttir

    source = root / "source.ttir"
    pointer = "*fp16" if dtype == "f16" else "*bf16"
    source.write_text(
        capture_ttir(
            half_atomic_add,
            {
                "x": pointer,
                "index": "*i32",
                "state": pointer,
                "old": "*i16",
                "n": "i32",
                "m": "i32",
                "B": "constexpr",
                "SCALAR": "constexpr",
            },
            {"B": 8, "SCALAR": scalar},
        )
    )
    abi = TensorABI(
        "half_atomic_add",
        (
            TensorBuffer("x", ("n",), dtype=dtype),
            TensorBuffer("index", ("n",), dtype="i32"),
            TensorBuffer("state", ("m",), "readwrite", dtype),
            TensorBuffer("old", ("n",), "write", "i16"),
        ),
        grid=(GridExtent("n", 1 if scalar else 8),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
            RuntimeScalar("m", "i32", 5, bounds=(0, 37)),
        ),
    )
    return convert_shared(
        source,
        abi,
        root / "translation",
        optimizer=OPT,
        memory_profile="guarded-buffer",
        address_policy="source-words",
        effect_profile="typed-atomics",
    )


def arguments(dtype, n=31, rotation=0):
    from t2m.examples.float_cast_reference import add_word
    from t2m.examples.scalar_math_reference import arrays

    if dtype == "f16":
        states = [
            0,
            0x8000,
            1,
            0x8001,
            0x0400,
            0x03FF,
            0x3C00,
            0x3C01,
            0x7BFF,
            0x7C00,
            0xFC00,
            0x7C01,
            0xFE35,
        ]
        values = [
            0x8000,
            0x8000,
            1,
            0x8001,
            0x83FF,
            1,
            0x1000,
            0x1000,
            0x7BFF,
            0xFC00,
            0x7C00,
            0x3C00,
            0x7C01,
        ]
    else:
        states = [
            0,
            0x8000,
            1,
            0x8001,
            0x0080,
            0x007F,
            0x3F80,
            0x3F81,
            0x7F7F,
            0x7F80,
            0xFF80,
            0x7F81,
            0xFFE5,
        ]
        values = [
            0x8000,
            0x8000,
            1,
            0x8001,
            0x807F,
            1,
            0x3B80,
            0x3B80,
            0x7F7F,
            0xFF80,
            0x7F80,
            0x3F80,
            0x7F81,
        ]
    incoming = np.resize(np.roll(np.asarray(states, np.uint16), rotation), 37)
    increments = np.resize(np.roll(np.asarray(values, np.uint16), rotation), n)
    index = np.arange(n, dtype=np.int32)
    enabled = index % 3 != 1
    index[~enabled] = -8191
    updated, old = incoming.copy(), np.zeros(n, np.uint16)
    for lane in range(n):
        if enabled[lane]:
            old[lane] = updated[index[lane]]
            updated[index[lane]] = add_word(
                int(updated[index[lane]]), int(increments[lane]), dtype
            )
    return [
        arrays(dtype, increments),
        index,
        arrays(dtype, incoming),
        np.asarray(n, np.int32),
        np.asarray(37, np.int32),
    ], [arrays(dtype, updated), old.view(np.int16)]


def check(actual, expected, inputs, dtype):
    from t2m.examples.scalar_math_reference import fields

    assert actual[1].tobytes() == expected[1].tobytes()
    _, fraction, _, exponent, _ = fields(dtype)
    fraction_mask, quiet = (1 << fraction) - 1, 1 << (fraction - 1)
    written = np.zeros(expected[0].shape, np.bool_)
    lanes = np.arange(len(inputs[1]))
    written[inputs[1][lanes % 3 != 1]] = True
    for changed, got, want in zip(
        written, actual[0].view(np.uint16), expected[0].view(np.uint16), strict=True
    ):
        nan = (int(want) & exponent) == exponent and (int(want) & fraction_mask) != 0
        if changed and nan:
            assert int(got) & exponent == exponent
            assert int(got) & fraction_mask
            assert int(got) & quiet
        else:
            assert got == want


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("scalar", [False, True])
def test_native_half_add_uses_exact_returned_old_words(tmp_path, dtype, scalar):
    from t2m.native import execute_native

    result = capture(tmp_path, dtype, scalar)
    for index, (n, rotation) in enumerate(((0, 0), (1, 0), (31, 0), (31, 1), (31, 2))):
        values, expected = arguments(dtype, n, rotation)
        actual, _ = execute_native(
            result.mlir_text,
            values,
            tmp_path / f"native-{index}",
            python=NATIVE,
            shape_abi=result.spec.runtime_shapes(),
        )
        check(actual, expected, values, dtype)
    # Collisions have several valid orders. Check the complete old-value history
    # instead of requiring a particular parallel lane order.
    from t2m.examples.scalar_math_reference import arrays

    one = 0x3C00 if dtype == "f16" else 0x3F80
    indices = np.arange(31, dtype=np.int32) % 3
    active = np.arange(31) % 3 != 1
    indices[~active] = -8191
    inputs = [
        arrays(dtype, np.full(31, one, np.uint16)),
        indices,
        arrays(dtype, np.zeros(37, np.uint16)),
        np.asarray(31, np.int32),
        np.asarray(37, np.int32),
    ]
    actual, _ = execute_native(
        result.mlir_text,
        inputs,
        tmp_path / "collisions",
        python=NATIVE,
        shape_abi=result.spec.runtime_shapes(),
    )
    from t2m.tensor_storage import comparison_values

    state = comparison_values(actual[0], dtype)
    old = comparison_values(arrays(dtype, actual[1].view(np.uint16)), dtype)
    for target in (0, 2):
        selected = active & (indices == target)
        assert state[target] == np.count_nonzero(selected)
        np.testing.assert_array_equal(
            np.sort(old[selected]), np.arange(np.count_nonzero(selected))
        )
    assert np.count_nonzero(state) == 2 and np.all(old[~active] == 0)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("scalar", [False, True])
def test_merlin_half_add_retains_all_words_and_reuses_dynamic_library(
    tmp_path, dtype, scalar
):
    import json
    from test_mixed_effects import expected as effect_reference, qualify
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    result = capture(tmp_path, dtype, scalar)
    info = result.trace["memory"]["atomic_payload"][0]
    assert info["numeric_contract"] == f"ieee-{dtype}-rne-add-quiet-nans"
    assert (
        "memref.generic_atomic_rmw" in result.mlir_text
        and "llvm." not in result.mlir_text
    )
    inputs, reference = arguments(dtype)
    policy = effect_reference(
        "", [{"kind": "values", "rtol": 0, "atol": 0}, {"kind": "words"}]
    )
    bundle, compiled = qualify(result, tmp_path, inputs, policy, reference)
    assert (
        "IEEE f16/BF16 atomic addition" in (bundle / "compatibility.json").read_text()
    )
    library = _record(compiled / "host/model.so")
    with np.load(compiled / "evaluation/outputs.npz", allow_pickle=False) as actual:
        check([actual["out0"], actual["out1"]], reference, inputs, dtype)
    for index, (n, rotation) in enumerate(((0, 0), (1, 0), (31, 1), (31, 2))):
        values, reference = arguments(dtype, n, rotation)
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(caller / "inputs.npz", **{f"in{i}": a for i, a in enumerate(values)})
        np.savez(caller / "reference.npz", out0=reference[0], out1=reference[1])
        (caller / "reference.json").write_text(json.dumps(policy))
        replay = tmp_path / f"replay-{index}"
        invoke_kernel(
            compiled,
            caller / "inputs.npz",
            caller / "reference.json",
            replay,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(replay)
        assert _record(replay / "source/host/model.so") == library
        with np.load(replay / "evaluation/outputs.npz", allow_pickle=False) as actual:
            check([actual["out0"], actual["out1"]], reference, values, dtype)
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_effect as target

        options = dict(
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        target.qualify(bundle, tmp_path / "target", **options)
        target.verify(tmp_path / "target", **options)
        inputs, reference = arguments(dtype)
        with np.load(tmp_path / "target/outputs.npz", allow_pickle=False) as actual:
            check([actual["out0"], actual["out1"]], reference, inputs, dtype)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(
    "change",
    [
        None,
        "width",
        "fractional-width",
        "format",
        "rounding",
        "payload",
        "storage",
        "kind",
        "profile",
    ],
)
def test_half_atomic_evidence_requires_actual_type_and_numeric_contract(dtype, change):
    from copy import deepcopy
    from test_float_atomic_add import evidence
    from t2m.atomic_memory import validate_evidence

    admission, memory, records, buffers = evidence(strict=True)
    info = admission["accesses"][0]["atomic"]
    info.update(
        word_bits=16,
        value_dtype=dtype,
        numeric_contract=f"ieee-{dtype}-rne-add-quiet-nans",
    )
    buffers["state"]["dtype"] = dtype
    admission["atomic_adaptations"][0]["payload"] = (
        f"standard-memref-generic-{dtype}-quiet-add-with-original-returned-old-value"
    )
    if change == "width":
        info["word_bits"] = 32
    elif change == "fractional-width":
        info["word_bits"] = 16.0
    elif change == "format":
        info["numeric_contract"] = (
            f"ieee-{'bf16' if dtype == 'f16' else 'f16'}-rne-add-quiet-nans"
        )
    elif change == "rounding":
        info["numeric_contract"] = "round-toward-zero"
    elif change == "payload":
        admission["atomic_adaptations"][0]["payload"] = "unqualified-add"
    elif change == "storage":
        buffers["state"]["dtype"] = "i16"
    elif change == "kind":
        info.update(operation="exch", standard_kind=3)
    elif change == "profile":
        admission.update(
            schema="t2m.memory-admission.v3",
            effect_profile="private-invocation-integer-atomics-v1",
        )
        memory["effect_profile"] = admission["effect_profile"]
    records[0]["atomic"] = deepcopy(info)
    admission["atomic_adaptations"][0].update(info)
    memory["atomic_payload"] = deepcopy(admission["atomic_adaptations"])
    if change is None:
        validate_evidence(admission, memory, records, buffers)
    else:
        with pytest.raises(ValueError, match="atomic memory"):
            validate_evidence(admission, memory, records, buffers)
