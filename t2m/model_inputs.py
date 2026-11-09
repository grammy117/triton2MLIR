"""Private subprocess: materialize a model2MLIR manifest using safetensors."""

import hashlib
import json
from pathlib import Path
import resource
import sys

import numpy as np
from safetensors import safe_open

DTYPES = {"f16": np.float16, "f32": np.float32, "f64": np.float64, "i64": np.int64,
          "i32": np.int32, "i16": np.int16, "i8": np.int8, "i1": np.bool_}


def main(capture, work):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_DATA, (512 * 1024 * 1024, 512 * 1024 * 1024))
    capture, work = Path(capture), Path(work)
    expected = json.loads((work / "abi.json").read_text())
    manifest = json.loads((capture / "weights.safetensors.manifest.json").read_text())
    order = json.loads((capture / "input_order.json").read_text())
    if set(manifest) != {str(i) for i in range(len(expected))}:
        raise ValueError("model manifest does not cover the complete forward ABI")
    arrays, sources = [], []
    with safe_open(capture / "weights.safetensors", framework="np") as weights, \
            np.load(capture / "inputs.npz", allow_pickle=False) as inputs, \
            np.load(capture / "extra.npz", allow_pickle=False) as extra:
        for i, abi in enumerate(expected):
            meta = manifest[str(i)]
            kind, name = meta.get("kind"), meta.get("weight", meta.get("name"))
            if kind in ("param", "buffer") and name in weights.keys():
                array, storage = weights.get_tensor(name), f"weights:{name}"
            elif kind == "buffer":
                candidates = [k for k in extra.files if k.startswith("buf::") and
                              "b_" + k[5:].replace(".", "_") == name]
                if len(candidates) != 1:
                    raise ValueError(f"buffer {name!r} has no unique extra.npz binding")
                array, storage = extra[candidates[0]], f"extra:{candidates[0]}"
            elif isinstance(name, str) and name.startswith("c_lifted_tensor_") and name in extra:
                array, storage = extra[name], f"extra:{name}"
            elif kind == "input" and name in order and type(order[name]) is int:
                key = f"in{order[name]}"
                array, storage = inputs[key], f"inputs:{key}"
            else:
                raise ValueError(f"argument {i} has no supported exact manifest binding")
            if tuple(array.shape) != tuple(abi["shape"]) or array.dtype != DTYPES[abi["dtype"]]:
                raise ValueError(f"materialized argument {i} differs from its forward type")
            array = np.ascontiguousarray(array).copy()
            arrays.append(array)
            sources.append({"argument": i, "storage": storage,
                            "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
    np.savez(work / "arguments.npz", **{f"in{i}": a for i, a in enumerate(arrays)})
    (work / "bindings.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
