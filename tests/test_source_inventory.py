"""Original native source remains measurable even without a product conversion."""

import importlib.util
import json

import pytest

from t2m.diagnostics import ConversionError
from t2m.frontend import inspect_ttir

pytestmark = pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="pinned Triton required")


def test_native_inventory_keeps_pointer_observations_and_original_ssa(tmp_path):
    source = tmp_path / "source.ttir"
    original = '''module {
      tt.func public @copy(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>) {
        %bits=tt.ptr_to_int %x:!tt.ptr<f32> -> i64
        %p=tt.int_to_ptr %bits:i64 -> !tt.ptr<f32>
        %v=tt.load %p:!tt.ptr<f32>
        tt.store %out,%v:!tt.ptr<f32>
        tt.return
      }
    }'''
    source.write_text(original)
    result = inspect_ttir(source, tmp_path / "inspection")
    assert result["normalization"] == "not-performed" and result["target_backend"] == "not-invoked"
    assert result["triton_version"] == "3.7.1"
    assert (tmp_path / "inspection/source.ttir").read_bytes() == source.read_bytes()
    operations = result["operations"]
    first, second = (next(op for op in operations if op["opcode"] == name)
                     for name in ("tt.ptr_to_int", "tt.int_to_ptr"))
    assert first["outputs"] == second["inputs"]
    assert first["output_types"] == second["input_types"] == ["i64"]
    assert len({op["id"] for op in operations}) == len(operations)
    assert result["blocks"][0]["arguments"][0]["type"] == "!tt.ptr<f32>"
    assert json.loads((tmp_path / "inspection/inventory.json").read_text()) == result


@pytest.mark.parametrize("body", ('"tt.future_operation"() : () -> ()',
    '%v=tt.load %x:!tt.ptr<f32>\ntt.store %out,%v:!tt.ptr<i32>'))
def test_native_invalid_source_retains_bytes_and_diagnostic_without_success_inventory(tmp_path, body):
    source = tmp_path / "invalid.ttir"
    source.write_text(f'module {{ tt.func @bad(%x:!tt.ptr<f32>,%out:!tt.ptr<f32>) {{ {body}\ntt.return }} }}')
    work = tmp_path / "inspection"
    with pytest.raises(ConversionError, match="source verification failed"):
        inspect_ttir(source, work)
    assert (work / "source.ttir").read_bytes() == source.read_bytes()
    assert (work / "stderr.log").stat().st_size > 0
    assert not (work / "inventory.json").exists()
