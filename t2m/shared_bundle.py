"""Package the shared tensor profile after actual native numerical validation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from .bundle import _json, _record
from .native import execute_native
from .shared_tensor import TensorABI
from .memory_contract import BUFFER_PROFILE, GUARDED_PROFILES, WORD_PROFILE, PROFILE_FILES, TENSOR_PROFILE, validate_memory_evidence
from .runtime_shapes import resolve_shapes

REQUIRED = {"model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
            "inputs.npz", "golden.npy", "extra.npz", "input_order.json", "source.ttir",
            "spec.json", "provenance.json", "compatibility.json", "meta.json", "execution.json"}
DIFFERENCES = [
    "t2m.capture-receipt.v2 and TTIR provenance; not the model2MLIR FX receipt/trace schemas",
    "provenance v2 separates original native TTIR inventory from normalized graph IDs when an official intake pass is required",
    "entry retains the kernel name; fresh write outputs omit input storage after initialization checks",
    "writable tensor results follow buffer declaration order; readwrite destinations retain incoming values",
    "scalar_constants specialize source integer arguments; they are not runtime tensor arguments",
    "ABI v3 runtime_scalars retain source positions and plain scalar MLIR inputs; inputs.npz stores typed zero-dimensional values passed by value",
    "ABI v4 bounded integer scalars add executable inclusive range guards and symbolic runtime prefix extents; logical allocation capacities remain static",
    "no captured learned parameters; weights.safetensors is a genuine empty container",
    "golden.npy is result zero; extra.npz contains out1, out2, ... for any other results",
    "numeric checks use a supplied reference and explicit tolerances; original Triton execution is not performed",
    "native host execution is measured; Merlin model-call binding and staged/device execution remain unqualified",
]


def write_shared_bundle(result, out, inputs, references, *, work, python=None, rtol=0, atol=0):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("reference comparison requires finite nonnegative tolerances")
    source, mlir = result.source_ttir, result.mlir_text
    if hashlib.sha256(mlir.encode()).hexdigest() != result.trace["mlir_sha256"]:
        raise ValueError("IR changed after conversion; regenerate its provenance")
    if hashlib.sha256(source.encode()).hexdigest() != result.trace["source_sha256"]:
        raise ValueError("source changed after conversion")
    profile = validate_memory_evidence(result.trace, result.auxiliary_artifacts)
    arrays, references = tuple(np.asarray(a) for a in inputs), tuple(np.asarray(a) for a in references)
    if len(references) != len(result.spec.outputs):
        raise ValueError("supply one independent reference per returned tensor")
    actual, execution = execute_native(mlir, arrays, work, python=python, shape_abi=result.spec.runtime_shapes())
    for computed, golden in zip(actual, references, strict=True):
        if computed.shape != golden.shape or computed.dtype != golden.dtype:
            raise ValueError("reference shapes/dtypes must exactly match results")
        np.testing.assert_allclose(computed, golden, rtol=rtol, atol=atol, equal_nan=True)
    execution.update({"reference_comparison": "passed", "rtol": rtol, "atol": atol,
                      "comparison": "numeric-allclose-equal-nan; no NaN-payload/signed-zero-bit claim"})
    trace = deepcopy(result.trace)
    trace["native_verification"] = "passed"
    trace["example_numeric_validation"] = execution
    # A few passing examples do not discharge all source numeric obligations.
    trace["numeric_equivalence"] = "example-reference-passed; universal-equivalence-not-established"
    out.mkdir(parents=True, exist_ok=False)
    (out / "model.mlir").write_text(mlir)
    (out / "source.ttir").write_text(source)
    (out / "weights.safetensors").write_bytes(struct.pack("<Q", 8) + b"{}      ")
    _json(out / "weights.safetensors.manifest.json", {str(i): {"kind": "input", "name": b.name}
                                                    for i, b in enumerate(result.spec.inputs)})
    np.savez(out / "inputs.npz", **{f"in{i}": a for i, a in enumerate(arrays)})
    np.save(out / "golden.npy", references[0], allow_pickle=False)
    np.savez(out / "extra.npz", **{f"out{i}": a for i, a in enumerate(references) if i})
    _json(out / "input_order.json", {b.name: i for i, b in enumerate(result.spec.inputs)})
    _json(out / "spec.json", result.spec.as_dict())
    _json(out / "provenance.json", trace)
    _json(out / "execution.json", execution)
    for name, payload in result.auxiliary_artifacts.items():
        (out / name).write_text(payload)
    differences = DIFFERENCES + (["explicit mixed tensor/memref payload with archived semantic witness and memory continuation; the public tensor/scalar ABI is retained"] if profile == BUFFER_PROFILE else [])
    if profile in GUARDED_PROFILES:
        differences.append("explicit guarded standard-memory payload with an independent runtime admission report; active bounds, read/write ownership, completed initialization and per-store lane uniqueness are executable; source-base gathers retain memory effects before optimization; grid-derived metadata cost is explicit; no pure tensor witness is asserted")
    if profile == WORD_PROFILE:
        differences.append("explicit source-word address policy retains source integer arithmetic before signed i64 pointer-offset accumulation; raw offset tensors replace structured pointer views; word-stage input/output evidence is retained")
    selected = [t for t in trace.get("transformations", []) if t.get("kind") == "selected-pointer-memory"]
    if selected:
        differences.append("finite mixed-base pointers use internal logical-buffer selectors paired with source-word offsets; predicated accesses stay at the original memory event; the public tensor/scalar ABI is unchanged")
        if any(t["missing_other_zero_refinement"] for t in selected):
            differences.append("inactive undefined source load values use the pinned upstream gather's zero refinement; this does not assert that inactive source memory was initialized or read")
    if result.spec.dynamic:
        differences.append("ABI v5 named runtime tensor dimensions and logical ceil-divided grids; dynamic result allocation uses scalar bindings; Merlin staging uses compile-signature v2")
    _json(out / "compatibility.json", {"schema": "t2m.compatibility.v2", "profile": profile,
          "baseline": "model2MLIR 035bb5a", "differences": differences,
          "same_core_filenames": True, "merlin_staged_execution": "not-certified"})
    _json(out / "meta.json", {"frontend": "triton-shared-with-t2m-tensor-ABI", "entry": result.spec.entry,
          "output_type": "functional-tensor-with-standard-memory" if profile in (BUFFER_PROFILE, *GUARDED_PROFILES) else "linalg-on-tensors", "execution": "native-host-LLVM-JIT",
          "merlin_staged_execution": "not-certified"})
    receipt = {"schema": "t2m.capture-receipt.v2", "profile": profile,
               "artifacts": {name: _record(out / name) for name in sorted(REQUIRED | PROFILE_FILES[profile])},
               "materialized_abi": {"complete": True, "inputs": len(arrays), "outputs": len(actual)},
               "source_closure_verified": False}
    _json(out / "capture_receipt.json", receipt)
    verify_shared_bundle(out)
    return out


def verify_shared_bundle(out):
    out = Path(out)
    receipt = json.loads((out / "capture_receipt.json").read_text())
    profile = receipt.get("profile", TENSOR_PROFILE)
    if receipt.get("schema") != "t2m.capture-receipt.v2" or profile not in PROFILE_FILES or set(receipt.get("artifacts", {})) != REQUIRED | PROFILE_FILES[profile]:
        raise ValueError("not a complete shared tensor capture receipt")
    for name, record in receipt["artifacts"].items():
        path = out / name
        if path.is_symlink() or not path.is_file() or _record(path) != record:
            raise ValueError("bundle integrity mismatch: " + name)
    spec = TensorABI.from_dict(json.loads((out / "spec.json").read_text()))
    trace = json.loads((out / "provenance.json").read_text())
    if validate_memory_evidence(trace, {name: (out / name).read_bytes() for name in PROFILE_FILES[profile]}) != profile:
        raise ValueError("receipt and provenance memory profiles differ")
    if trace["spec"] != json.loads(json.dumps(spec.as_dict())):
        raise ValueError("provenance differs from the packaged ABI")
    if trace["mlir_sha256"] != _record(out / "model.mlir")["sha256"] or trace["source_sha256"] != _record(out / "source.ttir")["sha256"]:
        raise ValueError("provenance does not bind exact IR/source bytes")
    execution = json.loads((out / "execution.json").read_text())
    if execution["input_sha256"] != trace["mlir_sha256"] or execution.get("reference_comparison") != "passed":
        raise ValueError("native numerical validation does not bind the packaged IR")
    expected = {b.name: i for i, b in enumerate(spec.inputs)}
    if json.loads((out / "input_order.json").read_text()) != expected:
        raise ValueError("input ordering differs from the ABI")
    with np.load(out / "inputs.npz", allow_pickle=False) as data:
        if set(data.files) != {f"in{i}" for i in expected.values()}:
            raise ValueError("incomplete packaged inputs")
        result_shapes = resolve_shapes(spec.runtime_shapes(), [data[f"in{i}"] for i in range(len(spec.inputs))]) if spec.dynamic else tuple(b.shape for b in spec.outputs)
        for i, buffer in enumerate(spec.inputs):
            if not spec.dynamic and tuple(data[f"in{i}"].shape) != buffer.shape:
                raise ValueError("packaged input shape differs from the ABI")
            if buffer in spec.runtime_scalars:
                dtype = {"f32": np.float32, "f64": np.float64, "i1": np.bool_, "i8": np.int8,
                         "i16": np.int16, "i32": np.int32, "i64": np.int64}[buffer.dtype]
                if data[f"in{i}"].dtype != dtype:
                    raise ValueError("packaged runtime scalar dtype differs from the ABI")
    results = spec.outputs
    primary = np.load(out / "golden.npy", allow_pickle=False)
    if tuple(primary.shape) != result_shapes[0]:
        raise ValueError("primary golden shape differs from the ABI")
    with np.load(out / "extra.npz", allow_pickle=False) as extra:
        if set(extra.files) != {f"out{i}" for i in range(1, len(results))}:
            raise ValueError("incomplete additional references")
        for i, buffer in enumerate(results[1:], 1):
            if tuple(extra[f"out{i}"].shape) != result_shapes[i]:
                raise ValueError("additional reference shape differs from the ABI")
    if receipt["materialized_abi"] != {"complete": True, "inputs": len(spec.inputs), "outputs": len(results)}:
        raise ValueError("receipt ABI differs from the resolved spec")
    return receipt
