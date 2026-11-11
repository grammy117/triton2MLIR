"""Original i8 parameters observe a typed allocation at real byte granularity."""

import numpy as np
import pytest
from test_descriptor_state import REQUIRED

pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing shared/native/Merlin environments required"
)


@pytest.mark.parametrize("order", ("little", "big"))
def test_byte_pointer_parameter_updates_disjoint_bytes_of_public_f64_words(
    tmp_path, order
):
    from test_mixed_effects import qualify

    from t2m.allocation_bindings import (
        PointerArgument,
        ScalarArgument,
        convert_allocations,
    )
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.byte_buffer_kernels import typed_byte_lanes
    from t2m.examples.byte_buffer_reference import policy
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer

    source = tmp_path / "source.ttir"
    source.write_text(
        capture_ttir(
            typed_byte_lanes,
            {
                "raw": "*i8",
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
        "byte_parameter_state",
        (
            TensorBuffer("state", ("capacity",), "readwrite", "f64"),
            TensorBuffer("out", ("n",), "write", "i8"),
        ),
        grid=(GridExtent("n", 4),),
        runtime_scalars=(
            RuntimeScalar("n", "i32", 2, bounds=(0, 13)),
            RuntimeScalar("capacity", "i32", 3, bounds=(0, 2)),
            RuntimeScalar("origin", "i32", 4, bounds=(0, 16)),
        ),
    )
    result = convert_allocations(
        source,
        abi,
        (
            PointerArgument("state"),
            PointerArgument("out"),
            *(ScalarArgument(name) for name in ("n", "capacity", "origin")),
        ),
        tmp_path / "binding",
        source_entry="typed_byte_lanes",
        memory_profile="guarded-buffer",
        address_policy="source-words",
        byte_order=order,
    )
    words = np.asarray([0x7FF0000000000001, 0xFFF0000000000001], np.uint64)
    memory = bytearray(b"".join(int(word).to_bytes(8, order) for word in words))
    output = np.arange(13, dtype=np.int8) ^ np.int8(0x5A)
    memory[1:14] = output.tobytes()
    expected = np.asarray(
        [int.from_bytes(memory[index : index + 8], order) for index in (0, 8)],
        np.uint64,
    ).view(np.float64)
    values = [
        words.view(np.float64),
        np.asarray(13, np.int32),
        np.asarray(2, np.int32),
        np.asarray(1, np.int32),
    ]
    qualify(result, tmp_path, values, policy(), [expected, output])
