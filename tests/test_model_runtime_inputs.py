"""Runtime archive protocol and actual C storage; fixtures are not FX captures."""

import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest

from t2m.bundle import _record
from t2m.model_runtime_inputs import ARCHIVE, RECORD, prepare, project, verify


def fixture(root, raw=None):
    capture = root / "capture"
    (capture / "parent").mkdir(parents=True)
    raw = np.asarray([0, 0x8000, 1, 0x7f81, 0xffa5] if raw is None else raw, dtype=np.uint16)
    legacy = np.array(raw.astype(np.uint32) << 16).view(np.float32)
    legacy[np.isnan(legacy)] = np.float32(np.nan)
    np.savez(capture / "inputs.npz", in0=legacy, in1=np.array([2**62 + 3, -7], np.int64))
    np.savez(capture / "tensor_words.npz", word0=raw, word1=np.array([0x7fc1], np.uint16))
    entry = {"archive": "inputs.npz", "key": "in0", "word_key": "word0", "dtype": "bf16",
        "storage_dtype": "uint16", "shape": list(raw.shape)}
    (capture / "parent/meta.json").write_text(json.dumps({"tensor_words": {
        "schema": "m2m.tensor-word-archives.v1", "path": "tensor_words.npz",
        "entries": [entry, {**entry, "archive": "goldens.npz", "key": "out0", "word_key": "word1", "shape": [1]}]}}))
    (capture / "capture_receipt.json").write_text(json.dumps({"artifacts": {
        "tensor_words.npz": _record(capture / "tensor_words.npz")}}))
    (capture / "input_order.json").write_text('{"x":0,"y":1}')
    (capture / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "input", "name": "y"}, "1": {"kind": "input", "name": "x"}}))
    signature = {"inputs": [{"dtype": "i64", "shape": [2]}, {"dtype": "bf16", "shape": list(raw.shape)}]}
    out = root / "target"
    out.mkdir()
    return capture, out, signature, raw


def test_projection_uses_named_original_input_words_and_leaves_other_arrays_exact(tmp_path, monkeypatch):
    capture, out, signature, raw = fixture(tmp_path)
    before = {str(p): _record(p) for p in capture.rglob("*") if p.is_file()}
    getitem = np.lib.npyio.NpzFile.__getitem__

    def input_only(archive, key):
        if key == "word1":
            raise AssertionError("runtime materialization must not read reference words")
        return getitem(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", input_only)
    path = prepare(capture, out, signature)
    assert path == verify(capture, out, signature) == out / ARCHIVE
    with np.load(path, allow_pickle=False) as saved, np.load(capture / "inputs.npz", allow_pickle=False) as original:
        assert set(saved.files) == {"in0", "in1"}
        np.testing.assert_array_equal(saved["in0"].view(np.uint32), raw.astype(np.uint32) << 16)
        assert saved["in1"].tobytes() == original["in1"].tobytes()
    record = json.loads((out / RECORD).read_text())
    assert record["arguments"] == [{"argument": 1, "archive_key": "in0", "dtype": "bf16", "shape": [5],
        "word_witness": {"artifact": "tensor_words.npz", "key": "word0", "dtype": "bf16"},
        "source_word_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}]
    assert not any("golden" in p for p in record["source_members"])
    assert before == {str(p): _record(p) for p in capture.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="new artifacts"):
        prepare(capture, out, signature)


@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_scalar_empty_and_matrix_inputs_keep_their_shapes(tmp_path, shape):
    capture, out, signature, raw = fixture(tmp_path, np.zeros(shape, np.uint16))
    prepare(capture, out, signature)
    assert verify(capture, out, signature) == out / ARCHIVE
    with np.load(out / ARCHIVE) as data:
        assert data["in0"].shape == shape


@pytest.mark.parametrize("ordinal", [None, True, -1, 2])
def test_bfloat_input_cannot_use_positional_or_invalid_binding_fallback(tmp_path, ordinal):
    capture, out, signature, _ = fixture(tmp_path)
    (capture / "input_order.json").write_text(json.dumps({"x": ordinal, "y": 1}))
    with pytest.raises(ValueError, match="input-order binding"):
        prepare(capture, out, signature)
    assert not (out / RECORD).exists()


@pytest.mark.parametrize("damage", ["word", "extra", "record", "unpaired"])
def test_replay_refuses_rehashed_projection_damage(tmp_path, damage):
    capture, out, signature, _ = fixture(tmp_path)
    prepare(capture, out, signature)
    if damage == "record":
        record = json.loads((out / RECORD).read_text())
        record["arguments"][0]["source_word_sha256"] = "0" * 64
        (out / RECORD).write_text(json.dumps(record))
    elif damage == "unpaired":
        (out / RECORD).unlink()
    else:
        with np.load(out / ARCHIVE) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        if damage == "word":
            arrays["in0"].view(np.uint32)[3] ^= np.uint32(0x10000)
        else:
            arrays["out0"] = np.array([42])
        np.savez(out / ARCHIVE, **arrays)
    with pytest.raises(ValueError, match="runtime input projection"):
        verify(capture, out, signature)


@pytest.mark.parametrize("shape", [[-1], [True], [1 << 30]])
def test_invalid_or_excessive_runtime_extents_refuse_before_materialization(tmp_path, shape):
    capture, out, signature, _ = fixture(tmp_path)
    signature["inputs"][1]["shape"] = shape
    with pytest.raises(ValueError, match="bounded static tensor"):
        prepare(capture, out, signature)
    assert not (out / ARCHIVE).exists()


def test_legacy_embedding_recovery_is_explicit_and_weights_do_not_select_projection(tmp_path):
    capture, out, signature, raw = fixture(tmp_path, [0, 0x8000, 1, 0x7f80])
    (capture / "tensor_words.npz").unlink()
    (capture / "parent/meta.json").write_text("{}")
    prepare(capture, out, signature)
    assert "tensor_words.npz" not in json.loads((out / RECORD).read_text())["source_members"]
    assert project(capture, signature)[1]["arguments"][0]["word_witness"]["encoding"] == "unique-binary32-embedding-without-NaNs"
    manifest = json.loads((capture / "weights.safetensors.manifest.json").read_text())
    manifest["1"] = {"kind": "param", "weight": "x"}
    (capture / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    assert project(capture, signature) is None


def test_every_bfloat_encoding_reaches_actual_merlin_c_storage(tmp_path):
    runtime = pytest.importorskip("merlin.llvmlower.c_runtime")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("existing C compiler required")
    capture, out, signature, raw = fixture(tmp_path, np.arange(65536, dtype=np.uint16))
    values, _ = project(capture, signature)
    assert runtime._storage_bytes(values["in0"], "bf16") == raw.tobytes()
    source = '#include <stdio.h>\nstatic const unsigned short values[]={' + runtime._embed_array(values["in0"], "bf16") + '};\n'
    source += 'int main(void) {return fwrite(values,1,sizeof(values),stdout)!=sizeof(values);}\n'
    executable = out / "all-bfloat-words"
    built = subprocess.run([compiler, "-O2", "-x", "c", "-", "-o", str(executable)],
        input=source, text=True, capture_output=True, timeout=20)
    assert built.returncode == 0, built.stderr
    actual = subprocess.run([str(executable)], capture_output=True, timeout=10, check=True)
    assert actual.stdout == raw.tobytes()
