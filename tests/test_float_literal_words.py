"""Literal NaNs retain their original words before host floating conversion."""

import os

import pytest

from t2m.ir import text
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared, parse_shared

FORMATS = [
    ("f16", 16, 0x7E00),
    ("bf16", 16, 0x7FC0),
    ("f32", 32, 0x7FC00000),
    ("f64", 64, 0x7FF8000000000000),
]
NATIVE, OPT = os.environ.get("T2M_NATIVE_PYTHON"), os.environ.get("TRITON_SHARED_OPT")


@pytest.mark.parametrize("dtype,width,nan", FORMATS)
@pytest.mark.parametrize("form", ["scalar", "dense", "encoded"])
@pytest.mark.parametrize(
    "change", ["canonical", "payload", "signaling", "negative-signaling"]
)
def test_original_literal_words_execute_without_changing_nan_sign_payload_or_quiet_bit(
    tmp_path, dtype, width, nan, form, change
):
    if not NATIVE:
        pytest.skip("existing native compiler required")
    word = nan if change == "canonical" else nan | 1 if change == "payload" else nan - 1
    if change == "negative-signaling":
        word |= 1 << (width - 1)
    words = [word, 1 << (width - 1), 1, word]
    if form == "scalar":
        body = f"%literal=arith.constant {hex(word)}:{dtype}\n%out=tensor.splat %literal:tensor<4x{dtype}>"
        words = [word] * 4
    else:
        literal = (
            "[" + ",".join(hex(v) for v in words) + "]"
            if form == "dense"
            else '"0x'
            + b"".join(v.to_bytes(width // 8, "little") for v in words).hex()
            + '"'
        )
        body = f"%out=arith.constant dense<{literal}>:tensor<4x{dtype}>"
    module = parse_shared(
        f"module {{func.func @forward()->tensor<4x{dtype}> {{{body} return %out:tensor<4x{dtype}>}}}}"
    )
    module.verify()
    assert "arith.bitcast" in text(module)
    from t2m.native import execute_native

    actual, _ = execute_native(text(module), [], tmp_path / "native", python=NATIVE)
    assert actual[0].tobytes() == b"".join(
        v.to_bytes(width // 8, "little") for v in words
    )


def source(dtype, word):
    return f"""module {{tt.func public @literal(%out:!tt.ptr<{dtype}>) {{
      %lane=tt.get_program_id x:i32
      %pointer=tt.addptr %out,%lane:!tt.ptr<{dtype}>,i32
      %value=arith.constant {hex(word)}:{dtype}
      tt.store %pointer,%value:!tt.ptr<{dtype}>
      tt.return
    }} }}"""


@pytest.mark.skipif(
    not (OPT and NATIVE), reason="existing shared/native tools required"
)
@pytest.mark.parametrize("dtype,width,nan", FORMATS)
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_original_ttir_literals_preserve_exact_source_inventory_and_native_words(
    tmp_path, dtype, width, nan, profile
):
    from t2m.native import execute_native

    word = (nan - 1) | (1 << (width - 1))
    path = tmp_path / "source.ttir"
    path.write_text(source(dtype, word))
    result = convert_shared(
        path,
        TensorABI("literal", (TensorBuffer("out", (4,), "write", dtype),), grid=(4,)),
        tmp_path / "translation",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    assert any(
        r.get("floating_literal_words", {}).get("words")
        == word.to_bytes(width // 8, "little").hex()
        for r in result.trace["source_operations"]
    )
    assert any(
        r["kind"] == "source-floating-literal-words"
        for r in result.trace["transformations"]
    )
    actual, _ = execute_native(result.mlir_text, [], tmp_path / "native", python=NATIVE)
    assert actual[0].tobytes() == word.to_bytes(width // 8, "little") * 4
