"""Complete source fields survive native generic printing without passes."""

import ctypes
import hashlib
import json

import pytest
from xdsl.dialects import builtin

from t2m.diagnostics import ConversionError
from t2m.frontend import (
    SOURCE_PRINT_FLAGS,
    _enable_generic_source_print,
    capture_ttir,
    read_ttir_generic,
)
from t2m.ir import text
from t2m.shared_tensor import opcode, parse_shared


@pytest.mark.parametrize("descriptor", [False, True])
def test_real_pointer_sources_keep_complete_attributes_and_caller_printers(
    tmp_path, descriptor
):
    from t2m.examples.tensor_kernels import blockptr_add, descriptor_add

    kernel = descriptor_add if descriptor else blockptr_add
    source = tmp_path / "source.ttir"
    signature = {"x": "*fp32", "out": "*fp32", "N": "constexpr", "B": "constexpr"}
    original = capture_ttir(kernel, signature, {"N": 53, "B": 16})
    source.write_text(original)
    report = read_ttir_generic(source, tmp_path / "first")
    assert report["flags"] == list(SOURCE_PRINT_FLAGS)
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["native_generic_roundtrip_verified"] is True
    assert report["normalization"] == "not-performed"
    module = parse_shared((tmp_path / "first/lowered.mlir").read_text())
    entry = next(op for op in module.walk() if opcode(op) == "tt.func")
    fields = builtin.DictionaryAttr(
        {
            "nested": builtin.ArrayAttr(
                [builtin.StringAttr("bytes\x00λ"), builtin.IntegerAttr(73, builtin.i64)]
            ),
            "words": builtin.DenseIntOrFPElementsAttr.from_list(
                builtin.TensorType(builtin.i8, [3]), [0, -128, -1]
            ),
        }
    )
    entry.attributes["audit.complete_fields"] = fields
    # A malformed source semantic field still must reach the real preflight;
    # source printing cannot discard it just because typed native getters fail.
    loaded = next(
        op for op in module.walk() if opcode(op) in ("tt.load", "tt.descriptor_load")
    )
    loaded.attributes["t2m.fp8_overflow"] = builtin.IntegerAttr(9, builtin.i32)
    source.write_text(text(module))
    second = read_ttir_generic(source, tmp_path / "second")
    restored = parse_shared((tmp_path / "second/lowered.mlir").read_text())
    restored_entry = next(op for op in restored.walk() if opcode(op) == "tt.func")
    restored_load = next(
        op for op in restored.walk() if opcode(op) in ("tt.load", "tt.descriptor_load")
    )
    assert restored_entry.attributes["audit.complete_fields"] == fields
    assert restored_load.attributes["t2m.fp8_overflow"] == builtin.IntegerAttr(
        9, builtin.i32
    )
    assert [opcode(op) for op in restored.walk()] == [
        opcode(op) for op in module.walk()
    ]
    assert second == json.loads((tmp_path / "second/print.json").read_text())
    assert capture_ttir(kernel, signature, {"N": 53, "B": 16}) == original


def test_unknown_native_printer_exports_refuse_before_ffi_calls(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())
    with pytest.raises(ConversionError, match="printer exports are unavailable"):
        _enable_generic_source_print("unused-native-library")


def test_invalid_native_source_retains_diagnostics_without_success_receipt(tmp_path):
    source = tmp_path / "source.ttir"
    source.write_text('module {"tt.future_op"():()->()}')
    with pytest.raises(ConversionError, match="source printing failed"):
        read_ttir_generic(source, tmp_path / "printed")
    assert not (tmp_path / "printed/print.json").exists()
    assert (tmp_path / "printed/source.ttir").read_bytes() == source.read_bytes()
    assert (tmp_path / "printed/stderr.log").stat().st_size
