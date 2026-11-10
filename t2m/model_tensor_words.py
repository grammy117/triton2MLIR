"""Read receipt-bound model2MLIR exact-word witnesses at the evaluator boundary.

Protocol: m2m.capture.tensor_words, m2m.tensor-word-archives.v1. Legacy archives
remain unchanged; only a matching logical BF16 ABI member consumes uint16 words.
"""

import json
import hashlib
from pathlib import Path

import numpy as np

FILENAME = "tensor_words.npz"


class TensorWords:
    def __init__(self, capture, *, metadata=None):
        self.capture = Path(capture)
        self.metadata = Path(metadata) if metadata is not None else self.capture / "meta.json"
        self.saved = None
        self.entries = {}
        self.recovered = {}

    def __enter__(self):
        metadata = json.loads(self.metadata.read_text()) if self.metadata.is_file() else {}
        pointer = metadata.get("tensor_words")
        path = self.capture / FILENAME
        if pointer is None and not path.exists():
            return self
        if (not isinstance(pointer, dict) or set(pointer) != {"schema", "path", "entries"}
                or pointer.get("schema") != "m2m.tensor-word-archives.v1" or pointer.get("path") != FILENAME
                or not path.is_file() or path.is_symlink()
                or not isinstance(pointer.get("entries"), list) or not pointer["entries"]):
            raise ValueError("model BF16 words require their exact parent archive declaration")
        for i, entry in enumerate(pointer["entries"]):
            if (not isinstance(entry, dict) or set(entry) != {"archive", "key", "word_key", "dtype", "storage_dtype", "shape"}
                    or not isinstance(entry["archive"], str) or entry["archive"] not in ("inputs.npz", "extra.npz", "golden.npy", "goldens.npz")
                    or entry["word_key"] != f"word{i}" or entry["dtype"] != "bf16" or entry["storage_dtype"] != "uint16"
                    or not isinstance(entry["shape"], list) or any(type(n) is not int or n < 0 for n in entry["shape"])
                    or (entry["key"] is None) != (entry["archive"] == "golden.npy")
                    or entry["key"] is not None and (not isinstance(entry["key"], str) or not entry["key"])):
                raise ValueError("invalid model BF16 archive binding")
            binding = entry["archive"], entry["key"]
            if binding in self.entries:
                raise ValueError("duplicate model BF16 archive binding")
            self.entries[binding] = entry
        receipt_path = self.capture / "capture_receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if receipt.get("artifacts", {}).get(FILENAME) != {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}:
            raise ValueError("model BF16 words are not bound by the exact capture receipt")
        self.saved = np.load(path, allow_pickle=False)
        if len(self.saved.files) != len(self.entries) or set(self.saved.files) != {e["word_key"] for e in self.entries.values()}:
            self.saved.close()
            self.saved = None
            raise ValueError("model BF16 word inventory differs from its parent declaration")
        return self

    def __exit__(self, *_):
        if self.saved is not None:
            self.saved.close()

    def get(self, archive, key, legacy, dtype, shape):
        entry = self.entries.get((archive, key))
        if dtype != "bf16":
            if entry is not None:
                raise ValueError("exact BF16 declaration differs from the logical model dtype")
            return legacy
        if entry is None or self.saved is None:
            # Legacy finite BF16 values (also signed zero/infinity) have one
            # exact binary32 embedding. Recover only that unique bit pattern;
            # NaN payloads and rounded/nonembedding values supply no such fact.
            if self.entries or legacy.dtype != np.float32 or tuple(legacy.shape) != tuple(shape):
                raise ValueError("BF16 model values require receipt-bound exact source words")
            bits = legacy.view(np.uint32)
            if np.any((bits & 0x7fffffff) > 0x7f800000) or np.any(bits & 0xffff):
                raise ValueError("legacy BF16 NaNs/nonembedding values require exact source words")
            self.recovered[(archive, key)] = {"artifact": archive, "key": key, "dtype": "bf16",
                "encoding": "unique-binary32-embedding-without-NaNs"}
            return (bits >> 16).astype(np.uint16)
        words = self.saved[entry["word_key"]]
        if (words.dtype != np.uint16 or tuple(words.shape) != tuple(shape) or list(words.shape) != entry["shape"]
                or legacy.dtype != np.float32 or legacy.shape != words.shape):
            raise ValueError("BF16 word storage differs from the model/legacy archive ABI")
        bits = words.astype(np.uint32) << 16
        nan = (bits & 0x7fffffff) > 0x7f800000
        if not np.array_equal(np.isnan(legacy), nan) or not np.array_equal(legacy.view(np.uint32)[~nan], bits[~nan]):
            raise ValueError("BF16 exact source words disagree with legacy numeric values")
        return words

    def witness(self, archive, key):
        entry = self.entries.get((archive, key))
        return ({"artifact": FILENAME, "key": entry["word_key"], "dtype": "bf16"} if entry is not None
                else self.recovered.get((archive, key)))
