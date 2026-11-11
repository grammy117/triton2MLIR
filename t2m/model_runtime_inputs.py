"""Project exact captured input words into Merlin's runtime-only value archive.

The C harness accepts BF16 values as binary32 containers and emits their BF16
words. An exact high-word embedding preserves every BF16 encoding, including
signaling NaNs; legacy numeric NaN payloads cannot supply those original words.
This projection contains inputs only, never result references or source weights.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .bundle import _record
from .model_tensor_words import FILENAME, TensorWords


ARCHIVE = "runtime-inputs.npz"
RECORD = "runtime-inputs.json"


def project(capture, signature):
    """Return a checked input projection, or None when no BF16 input needs one.

    The caller verifies the complete bound source bundle before using its
    signature/manifest. Replaying this function verifies the projection without
    executing a compiler or reading any output/reference value.
    """
    capture = Path(capture)
    manifest = json.loads((capture / "weights.safetensors.manifest.json").read_text())
    inputs = signature["inputs"]
    if set(manifest) != {str(i) for i in range(len(inputs))}:
        raise ValueError("runtime input manifest differs from the complete entry ABI")
    selected = [(i, spec, manifest[str(i)]) for i, spec in enumerate(inputs)
                if spec["dtype"] == "bf16" and manifest[str(i)].get("kind") == "input"]
    if not selected:
        return None
    if any(not isinstance(s["shape"], list) or any(type(n) is not int or n < 0 for n in s["shape"])
           for s in inputs) or sum(math.prod(s["shape"]) * 8 for s in inputs) > 64 << 20:
        raise ValueError("runtime input projection requires bounded static tensor arguments")
    order = json.loads((capture / "input_order.json").read_text())
    metadata = capture / "parent/meta.json"
    sources = ["inputs.npz", "input_order.json", "weights.safetensors.manifest.json", "parent/meta.json"]
    arguments = []
    with np.load(capture / "inputs.npz", allow_pickle=False) as saved, TensorWords(capture, metadata=metadata) as words:
        arrays = {k: saved[k].copy() for k in saved.files}
        for index, spec, meta in selected:
            name = meta.get("name")
            ordinal = order.get(name) if isinstance(name, str) else None
            if type(ordinal) is not int or ordinal < 0 or f"in{ordinal}" not in saved.files:
                raise ValueError("BF16 runtime input requires an explicit source input-order binding")
            key = f"in{ordinal}"
            raw = words.get("inputs.npz", key, saved[key], "bf16", spec["shape"])
            # Integer shifting creates an exact container; no floating conversion
            # or arithmetic may quiet a signaling NaN or choose a new payload.
            embedding = raw.astype(np.uint32)
            embedding <<= np.uint32(16)
            arrays[key] = embedding.view(np.float32)
            arguments.append({"argument": index, "archive_key": key, "dtype": "bf16", "shape": spec["shape"],
                "word_witness": words.witness("inputs.npz", key),
                "source_word_sha256": hashlib.sha256(raw.tobytes()).hexdigest()})
        if words.saved is not None:
            sources.append(FILENAME)
    return arrays, {"schema": "t2m.runtime-input-projection.v1", "encoding": "bf16-exact-binary32-embedding",
        "source_members": {name: _record(capture / name) for name in sources}, "arguments": arguments}


def prepare(capture, out, signature):
    value = project(capture, signature)
    if value is None:
        return Path(capture) / "inputs.npz"
    arrays, record = value
    out = Path(out)
    if any((out / name).exists() or (out / name).is_symlink() for name in (ARCHIVE, RECORD)):
        raise ValueError("runtime input projection must use new artifacts")
    np.savez(out / ARCHIVE, **arrays)
    (out / RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return out / ARCHIVE


def verify(capture, out, signature):
    """Re-derive every projected byte rather than trusting rehashed evidence."""
    out = Path(out)
    if not (out / RECORD).is_file():
        if (out / ARCHIVE).exists() or (out / ARCHIVE).is_symlink():
            raise ValueError("runtime input projection lacks its derivation record")
        return Path(capture) / "inputs.npz"
    value = project(capture, signature)
    if value is None or json.loads((out / RECORD).read_text()) != value[1]:
        raise ValueError("runtime input projection differs from original input bindings")
    with np.load(out / ARCHIVE, allow_pickle=False) as saved:
        if len(saved.files) != len(value[0]) or set(saved.files) != set(value[0]):
            raise ValueError("runtime input projection changed the source input roster")
        for key, expected in value[0].items():
            actual = saved[key]
            if (actual.dtype, actual.shape, actual.tobytes()) != (expected.dtype, expected.shape, expected.tobytes()):
                raise ValueError("runtime input projection differs from original source words")
    return out / ARCHIVE
