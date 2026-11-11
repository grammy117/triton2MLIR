"""Typed owning tensors keep exact words around width-changing source memory."""

import json

import numpy as np
import pytest
from test_descriptor_state import REQUIRED

from t2m.byte_buffer_demo import convert
from t2m.examples.byte_buffer_reference import policy
from t2m.examples.typed_storage_reference import typed_arguments
from t2m.typed_storage_contract import WIDTHS

pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


@pytest.mark.parametrize("dtype", tuple(WIDTHS))
@pytest.mark.parametrize("order", ("little", "big"))
def test_typed_owner_executes_byte_updates_and_preserves_other_words(
    tmp_path, dtype, order
):
    from test_byte_buffers import replay
    from test_mixed_effects import qualify

    result = convert(tmp_path / "source", root_dtype=dtype, byte_order=order)
    assert result.spec.buffers[0].dtype == dtype
    admission = json.loads(result.auxiliary_artifacts["memory-admission.json"])
    assert admission["spec"]["buffers"][0]["dtype"] == "i8"
    assert admission["spec"]["buffers"][0]["shape"] == ["capacity", WIDTHS[dtype]]
    profile = result.trace["typed_storage_transport"]
    assert profile["roots"][0]["logical_dtype"] == dtype
    assert "prov.word_transport" in result.mlir_text
    values, expected = typed_arguments(dtype, order)
    _, qualified = qualify(result, tmp_path, values, policy(), expected)
    signature = json.loads((qualified / "compiler/signature.json").read_text())
    base = admission["guard_storage_bytes_per_output_element"]
    assert signature["guard_storage_bytes_by_result_element"] == [
        base * WIDTHS[dtype],
        base,
    ]
    for index, options in enumerate(
        (
            {"n": 13, "steps": 3, "capacity": 256, "stride": 8},
            {
                "n": 13,
                "steps": 2,
                "capacity": 256,
                "origin": 180,
                "stride": -8,
                "advance": -1,
            },
            {"n": 0, "capacity": 0, "origin": 8191},
            {"n": 0, "capacity": 17, "origin": -8192},
            {"n": 13, "steps": 0, "capacity": 17, "origin": 8191},
        )
    ):
        values, expected = typed_arguments(dtype, order, **options)
        replay(tmp_path / f"replay-{index}", qualified, values, expected)


@pytest.mark.parametrize("dtype,order", (("bf16", "little"), ("f64", "big")))
def test_runtime_matrix_owners_keep_their_public_axes(tmp_path, dtype, order):
    from test_byte_buffers import replay
    from test_mixed_effects import qualify

    result = convert(
        tmp_path / "source",
        root_dtype=dtype,
        byte_order=order,
        root_shape=("capacity", 2),
    )

    def matrix_arguments(**options):
        rows = options.pop("capacity", 256)
        values, expected = typed_arguments(dtype, order, capacity=rows * 2, **options)
        values[0] = values[0].reshape(rows, 2)
        expected[0] = expected[0].reshape(rows, 2)
        values[3] = np.asarray(rows, np.int32)
        return values, expected

    values, expected = matrix_arguments()
    _, qualified = qualify(result, tmp_path, values, policy(), expected)
    for index, options in enumerate(
        ({"capacity": 128, "n": 13, "stride": 8}, {"capacity": 0, "n": 0})
    ):
        values, expected = matrix_arguments(**options)
        replay(tmp_path / f"matrix-{index}", qualified, values, expected)


@pytest.mark.parametrize("order", ("little", "big"))
def test_disjoint_bytes_in_one_public_word_have_independent_program_owners(
    tmp_path, order
):
    from test_mixed_effects import qualify

    from t2m.dynamic_abi import GridExtent
    from t2m.examples.byte_buffer_kernels import typed_byte_lanes
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = tmp_path / "source.ttir"
    source.write_text(
        capture_ttir(
            typed_byte_lanes,
            {
                "raw": "*fp32",
                "out": "*i8",
                "n": "i32",
                "capacity": "i32",
                "origin": "i32",
                "B": "constexpr",
            },
            {"B": 4},
        )
    )
    abi = TensorABI(
        "typed_byte_lanes",
        (
            TensorBuffer("raw", ("capacity",), "readwrite", "f32"),
            TensorBuffer("out", ("n",), "write", "i8"),
        ),
        grid=(GridExtent("n", 4),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 2, bounds=(0, 8)),
            RuntimeScalar("capacity", "i32", 3, bounds=(0, 2)),
            RuntimeScalar("origin", "i32", 4, bounds=(0, 8)),
        ),
    )
    result = convert_shared(
        source,
        abi,
        tmp_path / "translation",
        memory_profile="guarded-buffer",
        address_policy="source-words",
        byte_order=order,
    )
    incoming = np.asarray([0x7F800001, 0xFF800001], np.uint32)
    byte_state = bytearray(b"".join(int(word).to_bytes(4, order) for word in incoming))
    output = np.arange(7, dtype=np.int8) ^ np.int8(0x5A)
    byte_state[1:8] = output.tobytes()
    expected = np.asarray(
        [int.from_bytes(byte_state[index : index + 4], order) for index in (0, 4)],
        np.uint32,
    ).view(np.float32)
    values = [
        incoming.view(np.float32),
        np.asarray(7, np.int32),
        np.asarray(2, np.int32),
        np.asarray(1, np.int32),
    ]
    qualify(result, tmp_path, values, policy(), [expected, output])
