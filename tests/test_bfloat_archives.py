"""Exact-word protocol fixtures; these are not claimed FX model captures."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from t2m.model_tensor_words import FILENAME, TensorWords


def record(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fixture(root, *, materialize=False):
    arrays = [np.array([0x7f81, 1, 0x8000], np.uint16)]
    entries = [{"archive": "inputs.npz", "key": "in0", "word_key": "word0", "dtype": "bf16", "storage_dtype": "uint16", "shape": [3]}]
    if materialize:
        arrays.extend([np.array(0xffa5, np.uint16), np.array([[0, 1], [0x7f81, 0x8000]], np.uint16)])
        entries = [
            {**entries[0], "archive": "extra.npz", "key": "buf::cache"},
            {**entries[0], "word_key": "word1", "shape": []},
            {**entries[0], "archive": "extra.npz", "key": "c_lifted_tensor_0", "word_key": "word2", "shape": [2, 2]}]
    def legacy(words):
        return (words.astype(np.uint32) << 16).view(np.float32)
    np.savez(root / "inputs.npz", in0=legacy(arrays[1] if materialize else arrays[0]))
    np.savez(root / "extra.npz", **({"buf::cache": legacy(arrays[0]), "c_lifted_tensor_0": legacy(arrays[2])} if materialize else {}))
    np.savez(root / FILENAME, **{f"word{i}": value for i, value in enumerate(arrays)})
    (root / "meta.json").write_text(json.dumps({"tensor_words": {"schema": "m2m.tensor-word-archives.v1", "path": FILENAME, "entries": entries}}))
    (root / "capture_receipt.json").write_text(json.dumps({"schema": "m2m.capture-receipt.v1", "artifacts": {FILENAME: record(root / FILENAME)}}))
    return arrays


@pytest.mark.parametrize("change", [None, "unbound", "stale", "schema", "duplicate", "inventory", "storage", "shape", "values", "logical-dtype"])
def test_exact_words_require_pinned_typed_original_archive_binding(tmp_path, change):
    expected = fixture(tmp_path)[0]
    meta_path = tmp_path / "meta.json"
    meta = json.loads(meta_path.read_text())
    if change == "unbound": (tmp_path / "capture_receipt.json").write_text('{"artifacts":{}}')
    elif change == "stale": (tmp_path / FILENAME).write_bytes((tmp_path / FILENAME).read_bytes() + b"changed")
    elif change == "schema": meta["tensor_words"]["schema"] = "unknown"
    elif change == "duplicate": meta["tensor_words"]["entries"].append({**meta["tensor_words"]["entries"][0], "word_key": "word1"})
    elif change in ("inventory", "storage", "shape", "values"):
        actual = expected.copy()
        if change == "storage": actual = actual.astype(np.float32)
        elif change == "shape": actual = actual.reshape(1, 3)
        elif change == "values": actual[1] = 0x3f80
        np.savez(tmp_path / FILENAME, **{"other" if change == "inventory" else "word0": actual})
        (tmp_path / "capture_receipt.json").write_text(json.dumps({"artifacts": {FILENAME: record(tmp_path / FILENAME)}}))
    meta_path.write_text(json.dumps(meta))
    def run():
        with TensorWords(tmp_path) as words, np.load(tmp_path / "inputs.npz", allow_pickle=False) as legacy:
            result = words.get("inputs.npz", "in0", legacy["in0"], "f32" if change == "logical-dtype" else "bf16", [3])
            np.testing.assert_array_equal(result, expected)
    if change is None:
        run()
    else:
        with pytest.raises(ValueError):
            run()


@pytest.mark.parametrize("legacy", [np.array([np.nan], np.float32), np.array([1.001], np.float32)])
def test_legacy_nan_or_nonembedding_values_do_not_invent_bfloat_words(tmp_path, legacy):
    with TensorWords(tmp_path) as words:
        with pytest.raises(ValueError, match="exact source words"):
            words.get("inputs.npz", "in0", legacy, "bf16", [1])


def test_legacy_exact_embeddings_recover_unique_words_and_declare_the_derivation(tmp_path):
    expected = np.array([0, 0x8000, 1, 0x807f, 0x3f80, 0xbf80, 0x7f80, 0xff80], np.uint16)
    legacy = (expected.astype(np.uint32) << 16).view(np.float32)
    with TensorWords(tmp_path) as words:
        actual = words.get("inputs.npz", "in0", legacy, "bf16", expected.shape)
        np.testing.assert_array_equal(actual, expected)
        assert words.witness("inputs.npz", "in0")["encoding"] == "unique-binary32-embedding-without-NaNs"


def test_model_comparison_uses_bfloat_values_instead_of_unsigned_word_distance():
    from t2m.model_bundle import _compare_results
    from xdsl.dialects.builtin import BFloat16Type, TensorType
    with pytest.raises(AssertionError):
        _compare_results([np.array([0xbf80], np.uint16)], [np.array([0x3f80], np.uint16)], 1., 0., [TensorType(BFloat16Type(), [1])])


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native environment required")
def test_buffer_lifted_constant_and_scalar_input_materialize_exact_words(tmp_path):
    import t2m
    expected = fixture(tmp_path, materialize=True)
    work = tmp_path / "arguments"
    work.mkdir()
    (work / "abi.json").write_text(json.dumps([{"shape": list(a.shape), "dtype": "bf16"} for a in expected]))
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", 8) + b"{}      ")
    (tmp_path / "weights.safetensors.manifest.json").write_text(json.dumps({"0": {"kind": "buffer", "name": "b_cache"},
        "1": {"kind": "input", "name": "x"}, "2": {"kind": "input", "name": "c_lifted_tensor_0"}}))
    (tmp_path / "input_order.json").write_text('{"x":0}')
    script = Path(t2m.__file__).with_name("model_inputs.py")
    result = subprocess.run([os.environ["T2M_NATIVE_PYTHON"], str(script), str(tmp_path), str(work)], capture_output=True,
        text=True, timeout=30, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(work / "arguments.npz", allow_pickle=False) as actual:
        for i, value in enumerate(expected):
            assert actual[f"in{i}"].shape == value.shape
            np.testing.assert_array_equal(actual[f"in{i}"], value)
