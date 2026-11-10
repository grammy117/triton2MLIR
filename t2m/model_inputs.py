"""Private subprocess: materialize a model2MLIR manifest using safetensors."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys

import numpy as np
from safetensors import safe_open
if __package__:
    from .model_tensor_words import TensorWords
    from .tensor_storage import DTYPES
else:
    from model_tensor_words import TensorWords
    from tensor_storage import DTYPES


def bfloat_weight(capture, argument, abi):
    root = os.environ.get("T2M_MERLIN_ROOT")
    if not root:
        raise ValueError("BF16 weight materialization requires the explicit T2M_MERLIN_ROOT reader path")
    sys.path.insert(0, str(Path(root).resolve() / "src"))
    from merlin.runtime.captured_constants import verify_capture_constant
    receipt = json.loads((capture / "capture_receipt.json").read_text())["artifacts"]
    checked = verify_capture_constant(manifest_path=capture / "weights.safetensors.manifest.json",
        manifest_sha256=receipt["weights.safetensors.manifest.json"]["sha256"],
        safetensors_path=capture / "weights.safetensors", safetensors_sha256=receipt["weights.safetensors"]["sha256"],
        entry_argument_index=argument, source_shape=abi["shape"], source_dtype="bf16", max_payload_bytes=64 * 1024 * 1024)
    return np.frombuffer(checked.logical_payload, dtype="<u2").reshape(abi["shape"]), checked.to_evidence()


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
    with TensorWords(capture) as words, safe_open(capture / "weights.safetensors", framework="np") as weights, \
            np.load(capture / "inputs.npz", allow_pickle=False) as inputs, \
            np.load(capture / "extra.npz", allow_pickle=False) as extra:
        for i, abi in enumerate(expected):
            evidence, witness = None, None
            meta = manifest[str(i)]
            kind, name = meta.get("kind"), meta.get("weight", meta.get("name"))
            if kind in ("param", "buffer", "constant") and name in weights.keys():
                storage = f"weights:{name}"
                if abi["dtype"] == "bf16":
                    array, evidence = bfloat_weight(capture, i, abi)
                else:
                    array = weights.get_tensor(name)
            elif kind == "buffer":
                candidates = [k for k in extra.files if k.startswith("buf::") and
                              "b_" + k[5:].replace(".", "_") == name]
                if len(candidates) != 1:
                    raise ValueError(f"buffer {name!r} has no unique extra.npz binding")
                array, storage = extra[candidates[0]], f"extra:{candidates[0]}"
                array = words.get("extra.npz", candidates[0], array, abi["dtype"], abi["shape"])
                witness = words.witness("extra.npz", candidates[0])
            elif isinstance(name, str) and name.startswith("c_lifted_tensor_") and name in extra:
                array, storage = extra[name], f"extra:{name}"
                array = words.get("extra.npz", name, array, abi["dtype"], abi["shape"])
                witness = words.witness("extra.npz", name)
            elif kind == "input" and name in order and type(order[name]) is int:
                key = f"in{order[name]}"
                array, storage = inputs[key], f"inputs:{key}"
                array = words.get("inputs.npz", key, array, abi["dtype"], abi["shape"])
                witness = words.witness("inputs.npz", key)
            else:
                raise ValueError(f"argument {i} has no supported exact manifest binding")
            if tuple(array.shape) != tuple(abi["shape"]) or array.dtype != DTYPES[abi["dtype"]]:
                raise ValueError(f"materialized argument {i} differs from its forward type")
            array = np.array(array, copy=True, order="C")
            arrays.append(array)
            sources.append({"argument": i, "storage": storage,
                            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                            **({"word_witness": witness} if witness else {}),
                            **({"captured_constant": evidence} if evidence else {})})
    np.savez(work / "arguments.npz", **{f"in{i}": a for i, a in enumerate(arrays)})
    (work / "bindings.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
