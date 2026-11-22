"""Translate discovered source launches and derive checked model bindings."""

import hashlib
import importlib.util
import sys
from pathlib import Path

from .frontend import capture_ttir
from .model_binding import KernelBinding
from .shared_tensor import TensorABI, TensorBuffer, convert_shared
from .torch_launch_capture import SCHEMA


def translate_launches(record, work, *, optimizer=None, memory_profile="auto"):
    """No hand-authored KernelBinding table or custom-op semantic replacement.

    Python kernel modules are trusted author inputs, like ordinary Triton source.
    The record binds their file bytes; source/compiler equivalence is checked by
    the existing converter, independently of the discovery metadata.
    """
    if record.get("schema") != SCHEMA or not isinstance(record.get("calls"), list) or not record["calls"]:
        raise ValueError("automatic binding requires a nonempty discovered launch record")
    if len(record["calls"]) > 128:
        raise ValueError("automatic binding exceeds the bounded 128-call profile")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    bindings = []
    for ordinal, call in enumerate(record["calls"]):
        source = call["source"]
        path = Path(source["path"])
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("discovered Triton source changed before translation")
        name = "_t2m_automatic_source_" + source["sha256"]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError("discovered Triton source is not a loadable Python module")
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(name)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            kernel = getattr(module, source["function"])
            original = capture_ttir(kernel, call["signature"], call["constexprs"])
        finally:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("discovered Triton source changed during translation")
        root = work / f"call-{ordinal:04d}"
        root.mkdir()
        ttir = root / "source.ttir"
        ttir.write_text(original)
        abi = TensorABI(source["function"], tuple(TensorBuffer(**{**buffer, "shape": tuple(buffer["shape"])})
            for buffer in call["buffers"]), grid=tuple(call["grid"]))
        implementation = convert_shared(ttir, abi, root / "translation", optimizer=optimizer,
            memory_profile=memory_profile)
        implementation.trace["automatic_launch"] = {"schema": SCHEMA, "call": call["call"],
            "source": source, "discovery": record["discovery"], "tensor_values_used": False}
        bindings.append(KernelBinding(call["qualname"], implementation, operands=call["operands"],
            results=tuple(call["results"]), scalar_arguments=call["scalar_arguments"]))
    return bindings
