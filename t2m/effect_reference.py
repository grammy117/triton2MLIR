"""Independent expected outcomes for source effects; no numeric placeholder."""

import json
from pathlib import Path
import signal

import numpy as np

from .process_outcome import record
from .runtime_shapes import resolve_shapes
from .tensor_storage import DTYPES


def validate_reference(reference, trace=None):
    if (not isinstance(reference,dict) or set(reference) != {"schema","termination","stdout","stderr"}
            or reference["schema"] != "t2m.effect-reference.v1"
            or reference["termination"] not in ("return","assertion_failure")
            or any(not isinstance(reference[key],str) or len(reference[key].encode("utf-8")) > 4*1024*1024 for key in ("stdout","stderr"))):
        raise ValueError("effect reference requires explicit termination and exact bounded text streams")
    if trace is not None and reference["termination"] == "assertion_failure":
        assertions = [r["source_assertion"] for r in trace.get("normalized_operations",trace.get("source_operations",[])) if "source_assertion" in r]
        messages = {a.get("diagnostic_text","Assertion `"+a["message"]+"` failed\n") for a in assertions}
        if reference["stdout"] not in messages or reference["stderr"]:
            raise ValueError("expected assertion diagnostic must correspond to an actual source assertion")
    return reference


def validate_inputs(spec, arrays):
    arrays = tuple(np.asarray(a) for a in arrays)
    if len(arrays) != len(spec.inputs) or sum(a.nbytes for a in arrays) > 64*1024*1024:
        raise ValueError("effect arguments differ from the complete bounded input ABI")
    if spec.dynamic:
        resolve_shapes(spec.runtime_shapes(),arrays)
    for array,value in zip(arrays,spec.inputs,strict=True):
        if array.dtype != DTYPES[value.dtype] or not spec.dynamic and tuple(array.shape) != value.shape:
            raise ValueError("effect argument differs from its declared tensor/scalar type")
        if value in spec.runtime_scalars and value.bounds is not None and not value.bounds[0] <= array.item() <= value.bounds[1]:
            raise ValueError("effect scalar is outside its declared ABI bounds")
        if value.dtype == "i1" and np.any(array.view(np.uint8) > 1):
            raise ValueError("noncanonical Boolean storage requires a byte-aware input ABI")
    return arrays


def compare_outcome(directory, reference, *, backend, program, arguments, implementation, result_count):
    """Demand real invocation phase and an exact expected observed outcome.

    Setup crashes, timeouts, other signals and missing completion cannot be
    reclassified as a source assertion. Failed calls publish no result archive.
    """
    reference = validate_reference(reference)
    root = Path(directory)
    outcome = json.loads((root / "outcome.json").read_text())
    for stream in ("stdout","stderr"):
        if outcome.get(stream) != record(root / (stream+".bin")):
            raise ValueError("effect observation stream differs from its byte binding")
    if (outcome.get("schema") != "t2m.worker-outcome.v1" or outcome.get("timed_out") is not False
            or type(outcome.get("returncode")) is not int):
        raise ValueError("incomplete or timed-out effect observation")
    if not (root / "invocation-started.json").is_file():
        raise ValueError("worker stopped before its actual kernel invocation phase")
    marker = json.loads((root / "invocation-started.json").read_text())
    if marker != {"schema":"t2m.invocation-started.v1","backend":backend,
                  "program":program,"arguments":arguments,"implementation":implementation}:
        raise ValueError("effect outcome lacks its actual source/library/argument invocation phase")
    code = 0 if reference["termination"] == "return" else -int(signal.SIGABRT)
    if outcome["returncode"] != code or any((root / (key+".bin")).read_bytes() != reference[key].encode("utf-8") for key in ("stdout","stderr")):
        raise ValueError("actual effect outcome differs from its independent reference")
    if reference["termination"] == "return":
        if result_count:
            raise ValueError("returned tensor results require a numeric reference profile")
        with np.load(root / "outputs.npz",allow_pickle=False) as outputs:
            if outputs.files:
                raise ValueError("void effect invocation published unexpected tensor results")
        completion = root / ("execution.json" if backend == "native-jit" else "evaluation.json")
        completed = json.loads(completion.read_text())
        if backend == "native-jit" and (completed.get("native_mlir_verified") is not True or completed.get("llvm_lowering_verified") is not True):
            raise ValueError("returned effect invocation lacks its completed native lowering")
        if backend == "merlin-host" and (completed.get("invocations") != 1 or completed.get("functional_inputs_preserved") is not True
                or completed.get("outputs") != record(root / "outputs.npz")):
            raise ValueError("returned effect invocation lacks its completed readonly host call")
    elif any((root / name).exists() for name in ("outputs.npz","execution.json","evaluation.json")):
        raise ValueError("failed effect invocation cannot publish completed result evidence")
    return {"termination":reference["termination"],"reference_comparison":"passed","result_count":0,
            "declared_result_count":result_count,"outputs_published":reference["termination"] == "return"}
