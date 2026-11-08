"""Public conversion API. Translation status never implies consumer execution support."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import hashlib

from . import frontend
from .analysis import analyze
from .ir import emit, text


@dataclass
class ConversionResult:
    module: object
    source_ttir: str
    spec: object
    trace: dict
    frontend: str = "triton"
    path_taken: str = "ttir"
    output_type: str = "linalg-on-tensors"

    @property
    def mlir_text(self):
        return text(self.module)

    @property
    def ok(self):
        return True  # Failed conversions raise ConversionError; no partial-success result.


def convert(ttir_or_kernel, spec):
    source = (frontend.parse(ttir_or_kernel, spec) if isinstance(ttir_or_kernel, (str, Path))
              else frontend.capture(ttir_or_kernel, spec))
    analyzed = analyze(source, spec)
    module, emitted, transformations = emit(analyzed)
    trace = {"schema": "t2m.provenance.v1", "source": {"kind": source.origin,
            "sha256": source.digest, "entry": spec.entry}, "spec_sha256": spec.digest,
            "toolchain": {"triton": version("triton"), "xdsl": version("xdsl")},
            "intent": {"description": spec.intent, "origin": "caller-description",
                       "grants_semantics": False},
            "source_operations": analyzed.records, "emitted_operations": emitted,
            "transformations": transformations,
            "proofs": {"bounds": "proved", "output_partition": "proved",
                       "stored_values_defined": "proved-through-pointwise-definedness",
                       "independence": "proved-under-independent-allocation-precondition",
                       "index_overflow_absent": "proved"},
            "preconditions": ["input/output buffers have independent storage",
                              "tensor arguments/results represent the declared logical f32 views",
                              "read-only input values are stable for the duration of the launch"],
            "output_initialization": ("incoming-values-preserved" if any(b.effect == "readwrite" for b in spec.buffers)
                                      else "proved-complete-overwrite"),
            "consumer_status": "not-tested", "numeric_profile": "f32-source-operations-no-reassociation",
            "mlir": {"sha256": hashlib.sha256(text(module).encode()).hexdigest()}}
    return ConversionResult(module, source.text, spec, trace)

