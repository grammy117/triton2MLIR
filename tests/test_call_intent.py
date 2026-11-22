"""Source call-grouping hints retain bodies, effects and explicit provenance."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_entry_values import arguments, compile_public, scalar_program
from test_narrow_atomics import REQUIRED, invoke

from t2m.allocation_bindings import (
    ResultBinding,
    ScalarArgument,
    ValueArgument,
    convert_allocations,
)
from t2m.bundle import _record
from t2m.frontend import inspect_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")


def intent(result, name):
    reports = [row for row in result.trace["transformations"] if row["kind"] == "source-call-grouping-intent"]
    assert len(reports) == 1 and reports[0]["function"] == name
    assert reports[0]["source_noinline"] and reports[0]["grants_semantics"] is False
    assert reports[0]["source_ids"] and reports[0]["body_source_ids"]
    assert json.loads(result.module.attributes["prov.source_call_intent"].data) == reports
    assert not any(op.name == "tt.call" for op in result.module.walk())
    return reports


@pytest.mark.parametrize("profile", ("tensor", "buffer", "guarded-buffer"))
def test_original_noinline_counterexample_executes_through_all_memory_routes(tmp_path, profile):
    ledger = json.loads((Path(__file__).parent / "data/v1_generic_gaps.json").read_text())
    gap = next(g for g in (*ledger["gaps"], *ledger["closed_gaps"]) if g["id"] == "source-noinline-hints")
    source = gap["witnesses"][0]["source"]
    path = tmp_path / "source.ttir"
    path.write_text(source)
    inspect_ttir(path, tmp_path / "source-verification")
    abi = TensorABI("kernel", (TensorBuffer("x", (1,), dtype="i32"), TensorBuffer("out", (1,), "write", "i32")))
    result = convert_shared(path, abi, tmp_path / "conversion", memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    intent(result, "helper")
    original = next(row for row in result.trace["source_operations"] if row["opcode"] == "tt.func"
                    and row["properties"].get("sym_name") == '"helper"')
    assert original["attributes"]["noinline"] == "true"
    arrays, expected = [np.asarray([41], np.int32)], [np.asarray([42], np.int32)]
    product = compile_public(result, tmp_path, arrays, expected)
    library = _record(tmp_path / "compiled/host/model.so")
    for i, value in enumerate((41, -7, 0)):
        actual = invoke(tmp_path, [np.asarray([value], np.int32)], f"invoke-{i}")
        np.testing.assert_array_equal(actual[0], np.asarray([value + 1], np.int32))
    assert _record(tmp_path / "compiled/host/model.so") == library
    product.verify()


def test_noinline_helper_keeps_repeated_state_effects_and_every_returned_value(tmp_path):
    source = '''module {
      tt.func private @update(%state:!tt.ptr<i32>,%delta:i32)->i32 attributes {noinline=true} {
        %old=tt.load %state:!tt.ptr<i32>
        %next=arith.addi %old,%delta:i32
        tt.store %state,%next:!tt.ptr<i32>
        tt.return %old:i32
      }
      tt.func public @twice(%state:!tt.ptr<i32>,%first:!tt.ptr<i32>,%second:!tt.ptr<i32>,%delta:i32) {
        %a=tt.call @update(%state,%delta):(!tt.ptr<i32>,i32)->i32
        %b=tt.call @update(%state,%delta):(!tt.ptr<i32>,i32)->i32
        tt.store %first,%a:!tt.ptr<i32>
        tt.store %second,%b:!tt.ptr<i32>
        tt.return
      }
    }'''
    path = tmp_path / "source.ttir"
    path.write_text(source)
    inspect_ttir(path, tmp_path / "source-verification")
    abi = TensorABI("twice", (TensorBuffer("state", (1,), "readwrite", "i32"),
        TensorBuffer("first", (1,), "write", "i32"), TensorBuffer("second", (1,), "write", "i32")),
        runtime_scalars=(RuntimeScalar("delta", "i32", 3, bounds=(-3, 5)),))
    result = convert_shared(path, abi, tmp_path / "conversion", memory_profile="guarded-buffer", address_policy="source-words")
    intent(result, "update")
    def values(state, delta):
        return [np.asarray([state], np.int32), np.asarray(delta, np.int32)], [
            np.asarray([state + 2 * delta], np.int32), np.asarray([state], np.int32), np.asarray([state + delta], np.int32)]
    arrays, expected = values(7, 3)
    product = compile_public(result, tmp_path, arrays, expected)
    for i, (state, delta) in enumerate(((7, 3), (-7, -3), (4, 0), (0, 5))):
        arrays, expected = values(state, delta)
        before = [a.tobytes() for a in arrays]
        actual = invoke(tmp_path, arrays, f"invoke-{i}")
        for value, golden in zip(actual, expected, strict=True):
            assert value.tobytes() == golden.tobytes()
        assert before == [a.tobytes() for a in arrays]
    product.verify()


def test_noinline_hint_composes_with_fp8_value_ports_and_owned_replay(tmp_path):
    dtype = "f8E5M2"
    source, abi = scalar_program(dtype, 256)
    marker = f"tt.func private @combine(%x:{dtype},%y:{dtype})->{dtype} {{"
    assert source.count(marker) == 1
    source = source.replace(marker, marker[:-1] + "attributes {noinline=true} {")
    path = tmp_path / "source.ttir"
    path.write_text(source)
    inspect_ttir(path, tmp_path / "source-verification")
    result = convert_allocations(path, abi,
        (ValueArgument("x", "origin", per_program=True), ValueArgument("y", per_program=True), ScalarArgument("enabled")),
        tmp_path / "conversion", source_entry="values", result_bindings=(ResultBinding("original"), ResultBinding("computed")),
        memory_profile="guarded-buffer", address_policy="source-words")
    intent(result, "combine")
    arrays, expected = arguments(dtype, 256, True, 0)
    product = compile_public(result, tmp_path, arrays, expected, owned=True)
    for i, (n, enabled, shift) in enumerate(((256, True, 0), (256, False, 7), (13, True, 3), (1, True, 9), (0, False, 0))):
        arrays, expected = arguments(dtype, n, enabled, shift)
        actual = invoke(tmp_path, arrays, f"invoke-{i}")
        for value, golden in zip(actual, expected, strict=True):
            assert value.shape == golden.shape and value.tobytes() == golden.tobytes()
    product.verify()
