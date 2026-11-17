"""Model invocation shapes and storage bindings, separate from compiler facts.

Read array/container headers when reopening captured invocation shapes. An
explicit word roster additionally reads only those singleton source values.
These evaluator values never specialize portable MLIR or supply compiler facts.
"""

import json
import math
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCHEMA = "t2m.model-invocation-inputs.v2"
MAX_BYTES = 64 << 20


def decode_abi(value):
    dynamic = isinstance(value, dict)
    if dynamic:
        if set(value) != {"schema", "inputs"} or value["schema"] != SCHEMA:
            raise ValueError("dynamic model inputs require their explicit invocation profile")
        value = value["inputs"]
    if not isinstance(value, list) or any(not isinstance(row, dict) or set(row) != {"shape", "dtype"}
            or not isinstance(row["shape"], list) or not isinstance(row["dtype"], str)
            or any(not (type(size) is int and size >= 0 or dynamic and size is None) for size in row["shape"])
            for row in value):
        raise ValueError("model invocation inputs need complete ranked tensor types")
    return value, dynamic


def check_shape(declared, actual):
    if (len(declared) != len(actual) or any(type(size) is not int or not 0 <= size < 2**63 for size in actual)
            or any(expected is not None and expected != size for expected, size in zip(declared, actual, strict=True))):
        raise ValueError("materialized model argument differs from its ranked forward type")


def storage_binding(meta, order, weights, extra):
    kind, name = meta.get("kind"), meta.get("weight", meta.get("name"))
    if not isinstance(name, str):
        raise TypeError("model argument has no named storage binding")
    if kind in ("param", "buffer", "constant") and name in weights:
        return "weights", name
    if kind == "buffer":
        candidates = [key for key in extra if key.startswith("buf::") and "b_" + key[5:].replace(".", "_") == name]
        if len(candidates) != 1:
            raise ValueError("model buffer has no unique extra archive binding")
        return "extra.npz", candidates[0]
    if name.startswith("c_lifted_tensor_") and name in extra:
        return "extra.npz", name
    if kind == "input" and name in order and type(order[name]) is int and order[name] >= 0:
        return "inputs.npz", f"in{order[name]}"
    raise ValueError("model argument has no supported exact manifest binding")


def _npz_shapes(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.filename.endswith(".npy") or member.filename[:-4] in result:
                raise ValueError("model archive needs distinct NPY tensor entries")
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError("model array header version is unsupported")
                shape, _fortran, dtype = reader(stream, max_header_size=16384)
                size = math.prod(shape) * dtype.itemsize
                if (dtype.hasobject or any(type(n) is not int or n < 0 for n in shape)
                        or size > MAX_BYTES or stream.tell() + size != member.file_size):
                    raise ValueError("model array header exceeds its exact bounded storage")
            result[member.filename[:-4]] = shape
    return result


def captured_shapes(capture, abi, *, word_bindings=None):
    """Reopen invocation shapes and explicitly selected scalar storage words.

    All other argument values remain unopened. These are evaluator/runtime
    inputs, never a compiler value or a result-reference shape inference.
    """
    capture = Path(capture)
    expected, _dynamic = decode_abi(abi)
    manifest = json.loads((capture / "weights.safetensors.manifest.json").read_text())
    order = json.loads((capture / "input_order.json").read_text())
    if set(manifest) != {str(i) for i in range(len(expected))}:
        raise ValueError("model invocation manifest does not cover the public arguments")
    with (capture / "weights.safetensors").open("rb") as stream:
        length = stream.read(8)
        if len(length) != 8:
            raise ValueError("model weights lack their bounded safetensors header")
        length = struct.unpack("<Q", length)[0]
        if not 2 <= length <= 16 << 20 or length > (capture / "weights.safetensors").stat().st_size - 8:
            raise ValueError("model safetensors header exceeds its container")
        weights = json.loads(stream.read(length))
    if not isinstance(weights, dict):
        raise TypeError("model weights require a safetensors tensor dictionary")
    headers = {name: _npz_shapes(capture / name) for name in ("inputs.npz", "extra.npz")}
    shapes = []
    selected = set()
    if word_bindings is not None:
        if not isinstance(word_bindings, dict) or len(word_bindings) > 256:
            raise ValueError("captured size words require an explicit bounded binding roster")
        for binding in word_bindings.values():
            ordinal = binding.get("argument") if isinstance(binding, dict) else None
            if (type(ordinal) is not int or not 0 <= ordinal < len(expected)
                    or expected[ordinal]["dtype"] not in ("i8", "i16", "i32", "i64")
                    or any(type(size) is not int or size != 1 for size in expected[ordinal]["shape"])):
                raise ValueError("captured size words need exact singleton integer source storage")
            selected.add(ordinal)
    for index, row in enumerate(expected):
        archive, key = storage_binding(manifest[str(index)], order, weights, headers["extra.npz"])
        shape = weights[key].get("shape") if archive == "weights" else headers[archive].get(key)
        if not isinstance(shape, (tuple, list)):
            raise TypeError("captured model argument has no ranked storage header")
        check_shape(row["shape"], shape)
        if index not in selected:
            shapes.append(SimpleNamespace(shape=tuple(shape)))
            continue
        if archive != "weights":
            with np.load(capture / archive, allow_pickle=False) as arrays:
                shapes.append(arrays[key].copy())
        else:
            header = weights[key]
            offsets = header.get("data_offsets")
            width = int(row["dtype"][1:])
            if (header.get("dtype") != "I" + str(width) or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(offset) is not int for offset in offsets) or not 0 <= offsets[0] <= offsets[1]
                    or offsets[1]-offsets[0] != width//8
                    or length + 8 + offsets[1] > (capture / "weights.safetensors").stat().st_size):
                raise ValueError("captured size word differs from its bounded safetensors storage")
            with (capture / "weights.safetensors").open("rb") as stream:
                stream.seek(8 + length + offsets[0])
                raw = stream.read(width//8)
            shapes.append(np.frombuffer(raw, dtype=np.dtype("<i" + str(width//8))).astype(
                np.dtype("i" + str(width//8)), copy=True).reshape(shape))
    return shapes
