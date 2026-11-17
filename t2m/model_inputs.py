"""Private subprocess: materialize a model2MLIR manifest using safetensors."""

import hashlib
import json
import os
import resource
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

if __package__:
    from .model_input_abi import MAX_BYTES, check_shape, decode_abi, storage_binding
    from .model_tensor_words import TensorWords
    from .tensor_storage import DTYPES
else:
    from model_input_abi import MAX_BYTES, check_shape, decode_abi, storage_binding
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
    expected, dynamic = decode_abi(json.loads((work / "abi.json").read_text()))
    manifest = json.loads((capture / "weights.safetensors.manifest.json").read_text())
    order = json.loads((capture / "input_order.json").read_text())
    if set(manifest) != {str(i) for i in range(len(expected))}:
        raise ValueError("model manifest does not cover the complete forward ABI")
    arrays, sources = [], []
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    metadata = capture / "parent/meta.json" if receipt.get("schema") == "t2m.bound-model-receipt.v1" else None
    with TensorWords(capture, metadata=metadata) as words, safe_open(capture / "weights.safetensors", framework="np") as weights, \
            np.load(capture / "inputs.npz", allow_pickle=False) as inputs, \
            np.load(capture / "extra.npz", allow_pickle=False) as extra:
        for i, abi in enumerate(expected):
            evidence, witness = None, None
            meta = manifest[str(i)]
            archive, name = storage_binding(meta, order, weights.keys(), extra.files)
            if archive == "weights":
                storage = f"weights:{name}"
                if abi["dtype"] == "bf16":
                    actual = {**abi, "shape": list(weights.get_slice(name).get_shape())}
                    check_shape(abi["shape"], actual["shape"])
                    array, evidence = bfloat_weight(capture, i, actual)
                else:
                    array = weights.get_tensor(name)
            elif archive == "extra.npz":
                array, storage = extra[name], f"extra:{name}"
                check_shape(abi["shape"], array.shape)
                array = words.get("extra.npz", name, array, abi["dtype"], array.shape)
                witness = words.witness("extra.npz", name)
            else:
                array, storage = inputs[name], f"inputs:{name}"
                check_shape(abi["shape"], array.shape)
                array = words.get("inputs.npz", name, array, abi["dtype"], array.shape)
                witness = words.witness("inputs.npz", name)
            check_shape(abi["shape"], array.shape)
            if array.dtype != DTYPES[abi["dtype"]]:
                raise ValueError(f"materialized argument {i} differs from its forward type")
            array = np.array(array, copy=True, order="C")
            arrays.append(array)
            if sum(value.nbytes for value in arrays) > MAX_BYTES:
                raise ValueError("materialized model arguments exceed the 64MiB qualification budget")
            sources.append({"argument": i, "storage": storage,
                            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                            **({"shape": list(array.shape)} if dynamic else {}),
                            **({"word_witness": witness} if witness else {}),
                            **({"captured_constant": evidence} if evidence else {})})
    np.savez(work / "arguments.npz", **{f"in{i}": a for i, a in enumerate(arrays)})
    (work / "bindings.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
