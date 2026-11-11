"""Canonical quiet-NaN literals retain exact words across parser and native code."""

import os

import pytest

from t2m.ir import text
from t2m.shared_tensor import parse_shared

FORMATS = (
    ("f16", 16, 0x7E00),
    ("bf16", 16, 0x7FC0),
    ("f32", 32, 0x7FC00000),
    ("f64", 64, 0x7FF8000000000000),
)


@pytest.mark.parametrize("dtype,width,nan", FORMATS)
def test_canonical_nan_hex_and_encoded_bytes_round_trip(dtype, width, nan):
    words = [nan, 1 << (width - 1), 0, nan]
    expected = b"".join(word.to_bytes(width // 8, "little") for word in words)
    for literal in (
        "[" + ", ".join(hex(word) for word in words) + "]",
        '"0x' + expected.hex() + '"',
    ):
        module = parse_shared(
            f"module {{ %0 = arith.constant dense<{literal}> : tensor<4x{dtype}> }}"
        )
        for _ in range(3):
            assert next(iter(module.ops)).value.data.data == expected
            module = parse_shared(text(module))


@pytest.mark.parametrize("dtype,width,nan", FORMATS)
@pytest.mark.parametrize("change", ("payload", "signaling", "negative"))
def test_unqualified_nan_payloads_still_refuse_including_encoded_bytes(
    dtype, width, nan, change
):
    word = (
        nan | 1
        if change == "payload"
        else (nan - 1 if change == "signaling" else nan | (1 << (width - 1)))
    )
    for literal in (hex(word), '"0x' + word.to_bytes(width // 8, "little").hex() + '"'):
        with pytest.raises(
            Exception, match="NaN constant payload preservation is not qualified"
        ):
            parse_shared(
                f"module {{ %0 = arith.constant dense<{literal}> : tensor<4x{dtype}> }}"
            )


@pytest.mark.skipif(
    not os.environ.get("T2M_NATIVE_PYTHON"),
    reason="existing native compiler environment required",
)
@pytest.mark.parametrize("dtype,width,nan", FORMATS)
def test_canonical_nan_and_signed_zero_are_native_constant_words(
    tmp_path, dtype, width, nan
):
    from t2m.native import execute_native

    words = [nan, 1 << (width - 1), 0, nan]
    literal = "[" + ", ".join(hex(word) for word in words) + "]"
    module = parse_shared(f"""module {{
      func.func @forward() -> tensor<4x{dtype}> {{
        %0 = arith.constant dense<{literal}> : tensor<4x{dtype}>
        return %0 : tensor<4x{dtype}>
      }}
    }}""")
    (actual,), _ = execute_native(
        text(module), [], tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"]
    )
    assert actual.tobytes() == b"".join(
        word.to_bytes(width // 8, "little") for word in words
    )
