"""Captured shapes come from bounded input headers and exact storage bindings."""

import json
import struct
import zipfile

import numpy as np
import pytest

from t2m.model_input_abi import SCHEMA, captured_shapes, decode_abi


def example(root):
    root.mkdir()
    (root / "input_order.json").write_text('{"x":0}')
    (root / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "param", "weight": "weight"}, "1": {"kind": "input", "name": "x"}}))
    header = json.dumps({"weight": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\x11\x00\x00\x00")
    np.savez(root / "inputs.npz", in0=np.arange(7, dtype=np.int32))
    np.savez(root / "extra.npz")
    return {"schema": SCHEMA, "inputs": [{"shape": [1], "dtype": "i32"}, {"shape": [None], "dtype": "i32"}]}


def test_input_headers_resolve_public_positions_without_arrays_or_any_reference(tmp_path, monkeypatch):
    abi = example(tmp_path / "capture")

    def no_array_values(*_args, **_kwargs):
        raise AssertionError("shape reopening must not load any tensor values")

    monkeypatch.setattr(np, "load", no_array_values)
    actual = captured_shapes(tmp_path / "capture", abi)
    assert [value.shape for value in actual] == [(1,), (7,)]
    np.savez(tmp_path / "capture/inputs.npz", in0=np.arange(3, dtype=np.int32))
    assert [value.shape for value in captured_shapes(tmp_path / "capture", abi)] == [(1,), (3,)]


@pytest.mark.parametrize("damage", ["profile", "legacy", "static", "binding", "manifest", "header", "duplicate"])
def test_invalid_rank_storage_profile_and_header_cannot_supply_invocation_shape(tmp_path, damage):
    root = tmp_path / "capture"
    abi = example(root)
    if damage == "profile":
        abi["schema"] = "unknown"
    elif damage == "legacy":
        abi = abi["inputs"]
    elif damage == "static":
        abi["inputs"][0]["shape"] = [2]
    elif damage == "binding":
        (root / "input_order.json").write_text('{"x":true}')
    elif damage == "manifest":
        (root / "weights.safetensors.manifest.json").write_text('{}')
    elif damage == "header":
        (root / "weights.safetensors").write_bytes(struct.pack("<Q", 1 << 30))
    else:
        with zipfile.ZipFile(root / "inputs.npz") as archive:
            raw = archive.read("in0.npy")
        with zipfile.ZipFile(root / "inputs.npz", "w") as archive:
            archive.writestr("in0.npy", raw)
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("in0.npy", raw)
    with pytest.raises(ValueError):
        captured_shapes(root, abi)


def test_dynamic_invocation_profile_is_explicit_and_preserves_legacy_static_inputs():
    rows = [{"shape": [7], "dtype": "i32"}]
    assert decode_abi(rows) == (rows, False)
    assert decode_abi({"schema": SCHEMA, "inputs": rows}) == (rows, True)


def test_reference_dimensions_cannot_override_shapes_from_actual_input_headers(tmp_path):
    from xdsl.dialects.builtin import DYNAMIC_INDEX, TensorType, i32

    from t2m.model_bundle import _references

    root = tmp_path / "capture"
    abi = example(root)
    actual = captured_shapes(root, abi)[1].shape
    np.save(root / "golden.npy", np.arange(13, dtype=np.int32))
    with pytest.raises(ValueError):
        _references(root, {}, [TensorType(i32, [DYNAMIC_INDEX])], result_shapes=[actual])


def test_explicit_runtime_word_projection_reads_only_selected_source_storage(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    abi = example(root)
    load = np.load

    def no_array_payloads(*_args, **_kwargs):
        raise AssertionError("an unrelated argument payload was loaded")

    monkeypatch.setattr(np, "load", no_array_payloads)
    words = {"w0": {"argument": 0, "indices": [0], "dtype": "i32"}}
    values = captured_shapes(root, abi, word_bindings=words)
    assert values[0].dtype == np.dtype(np.int32) and values[0].item() == 17
    assert values[1].shape == (7,) and not hasattr(values[1], "dtype")
    monkeypatch.setattr(np, "load", load)
    np.savez(root / "inputs.npz", in0=np.asarray(9, np.int64))
    abi["inputs"][1] = {"shape": [], "dtype": "i64"}
    values = captured_shapes(root, abi, word_bindings={"w1": {"argument": 1, "indices": [], "dtype": "i64"}})
    assert values[1].item() == 9 and not hasattr(values[0], "dtype")
