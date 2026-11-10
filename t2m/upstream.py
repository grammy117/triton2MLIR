"""Run reusable triton-shared stages without claiming our tensor/package contract.

No native dependency is built or installed. These are inspectable middle-layer
artifacts, not complete triton2MLIR conversions or qualified numerical results.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .diagnostics import ConversionError

SHARED_REVISION = "17f0c234ce474bee334af36e351c62eca7852836"
TRITON_REVISION = "142586f90aac9b917af1081e4697b085027e13b2"


@dataclass(frozen=True)
class Stage:
    flags: tuple[str, ...]
    boundary: str


STAGES = {
    "source": Stage((), "native-verified source IR, without transformations"),
    "inline": Stage(("--inline", "--symbol-dce"),
                    "source helper inlining; isolated from downstream dialect extensions"),
    "tensor-ir": Stage(("--triton-to-structured=use-unsafe-mask=false", "--cse",
                        "--canonicalize", "--triton-to-unstructured",
                        "--triton-arith-to-linalg=pids-to-func-args=false tt-to-func-func=false",
                        "--remove-dead-values", "--canonicalize"),
                       "tensor computation and structured memory; logical program IDs retained"),
    "tensor-ir-data-control": Stage(("--triton-to-structured=use-unsafe-mask=false skip-prepass=true", "--cse",
                        "--canonicalize", "--triton-to-unstructured",
                        "--triton-arith-to-linalg=pids-to-func-args=false tt-to-func-func=false",
                        "--remove-dead-values", "--canonicalize"),
                       "proved numeric control state without pointer recurrence; structured memory conversion retained"),
    "source-word-tensor": Stage(("--triton-to-unstructured=offset-bit-width=64",
                        "--triton-arith-to-linalg=pids-to-func-args=false tt-to-func-func=false",
                        "--remove-dead-values", "--canonicalize"),
                       "original integer word arithmetic with signed i64 pointer-offset accumulation; logical program IDs retained"),
    "source-word-events": Stage(("--triton-to-unstructured=offset-bit-width=64",),
                       "isolated source memory events with signed i64 offsets; no parent control or arithmetic lowering"),
    "source-word-compute": Stage(("--triton-arith-to-linalg=pids-to-func-args=false tt-to-func-func=false",
                        "--remove-dead-values", "--canonicalize"),
                       "tensor computation after in-place memory-event expansion; no second pointer conversion"),
    "structured": Stage(("--triton-to-structured=use-unsafe-mask=false",
                         "--remove-dead-values", "--canonicalize"),
                        "structured pointers and source tensor/control-flow operations"),
    "tensor-compute": Stage(("--triton-to-structured=use-unsafe-mask=false", "--cse",
                             "--canonicalize", "--triton-to-unstructured",
                             "--triton-arith-to-linalg", "--remove-dead-values", "--canonicalize"),
                            "linalg tensor computation with project-specific memory operations and per-program ABI"),
    "memory-ir": Stage(("--triton-arith-to-linalg", "--structured-to-memref", "--unstructured-to-memref",
                         "--triton-ptr-to-memref", "--triton-to-ptr", "--reconcile-unrealized-casts",
                         "--reconcile-ptr-casts", "--remove-dead-values", "--cse", "--canonicalize"),
                        "standard memory continuation after tensor computation/numeric adaptation; counts then program IDs appended"),
    "memory-effects": Stage(("--triton-arith-to-linalg", "--structured-to-memref", "--unstructured-to-memref",
                         "--triton-ptr-to-memref", "--triton-to-ptr", "--reconcile-unrealized-casts",
                         "--reconcile-ptr-casts", "--remove-dead-values"),
                        "standard memory continuation before CSE/canonicalization; mutable tensor views require effect-preserving adaptation"),
    "full": Stage(("--triton-to-linalg-experimental",),
                  "upstream memory/ptr/memref middle layer; not the functional tensor ABI"),
}

# Apply a file-size bound in a clean child, before replacing it with the native
# optimizer. Avoid preexec_fn in a parent that may have native-library threads.
_WRAPPER = """
import os, resource, sys
limit = int(sys.argv[1])
resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.execv(sys.argv[2], sys.argv[2:])
"""


def _record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class Optimizer:
    """One hashed tool session for a corpus; refuse executable replacement.

    A session avoids rehashing a 478MB optimizer for every tiny fixture. File
    metadata is checked before/after every invocation. This is a local workflow
    integrity check, not an attestation against hostile filesystem mutation.
    """

    path: Path
    record: dict
    identity: tuple

    def check(self):
        if _identity(self.path) != self.identity:
            raise ConversionError("optimizer changed during this tool session")


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def inspect_optimizer(optimizer=None):
    if isinstance(optimizer, Optimizer):
        optimizer.check()
        return optimizer
    configured = str(optimizer or os.environ.get("TRITON_SHARED_OPT", "triton-shared-opt"))
    executable = shutil.which(configured)
    if executable is None:
        raise ConversionError("triton-shared-opt is unavailable; set TRITON_SHARED_OPT or --optimizer "
                              "to an existing build; no source build is started automatically")
    executable = Path(executable).resolve()
    identity = _identity(executable)
    tool = Optimizer(executable, _record(executable), identity)
    tool.check()
    return tool


def lower_with_shared(source, out, *, optimizer=None, stage="structured", timeout=30,
                      output_limit=16 * 1024 * 1024, expected_optimizer_sha256=None):
    """Run a fixed audited upstream stage with bounded resources and exact hashes.

    Reference source pins describe the surveyed pipeline, not an assertion that
    an arbitrary executable was built from those sources. That distinction stays
    explicit even when the caller supplies a matching binary content hash.
    """
    if stage not in STAGES:
        raise ConversionError(f"unknown upstream stage {stage!r}; choose {', '.join(STAGES)}")
    if not 0 < timeout <= 60 or not 1024 <= output_limit <= 64 * 1024 * 1024:
        raise ConversionError("upstream invocation requires a 0..60s timeout and 1KiB..64MiB output bound")
    tool = inspect_optimizer(optimizer)
    executable, binary = tool.path, tool.record
    if expected_optimizer_sha256 is not None and binary["sha256"] != expected_optimizer_sha256:
        raise ConversionError("triton-shared optimizer hash does not match the requested binary pin")
    source, out = Path(source), Path(out)
    if not source.is_file() or not 0 < source.stat().st_size <= 4 * 1024 * 1024:
        raise ConversionError("upstream input must be a nonempty file at most 4MiB")
    input_bytes = source.read_bytes()
    if not input_bytes.strip():
        raise ConversionError("upstream input is empty")
    out.mkdir(parents=True, exist_ok=False)
    saved = out / "source.ttir"
    saved.write_bytes(input_bytes)
    output, diagnostics = out / "lowered.mlir", out / "stderr.log"
    command = [str(executable), "--mlir-disable-threading", "--verify-each",
               "--mlir-print-op-generic", "--mlir-print-debuginfo", "--mlir-print-local-scope",
               *STAGES[stage].flags, str(saved.resolve())]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       TMPDIR=str(out.resolve()))
    error = None
    try:
        with output.open("xb") as stdout, diagnostics.open("xb") as stderr:
            process = subprocess.run([sys.executable, "-c", _WRAPPER, str(output_limit), *command],
                                     stdout=stdout, stderr=stderr, timeout=timeout, env=environment)
        if process.returncode != 0:
            error = f"triton-shared stage failed with exit code {process.returncode}"
        elif not output.read_bytes().strip():
            error = "triton-shared produced no output"
    except subprocess.TimeoutExpired:
        error = f"triton-shared stage exceeded {timeout}s"
    tool.check()
    if error:
        _write_json(out / "failure.json", {"stage": stage, "error": error,
                     "portable_conversion_complete": False, "command": command})
        with diagnostics.open("rb") as stream:
            stream.seek(max(0, diagnostics.stat().st_size - 4000))
            detail = stream.read().decode(errors="replace")
        raise ConversionError(error + (": " + detail if detail else ""))
    report = {
        "schema": "t2m.triton-shared-stage.v1", "stage": stage, "stage_completed": True,
        "boundary": STAGES[stage].boundary, "portable_conversion_complete": False,
        "tensor_abi_compatible": False, "source_correspondence": "not-established",
        "numeric_equivalence": "not-qualified", "execution": "not-performed",
        "reference_source": {"repository": "https://github.com/facebookincubator/triton-shared",
                             "shared_revision": SHARED_REVISION, "triton_revision": TRITON_REVISION},
        "optimizer": {"path": str(executable), **binary,
                      "binary_hash_pin_checked": expected_optimizer_sha256 is not None,
                      "source_revision_verified": False},
        "command": command, "limits": {"timeout_seconds": timeout, "output_bytes": output_limit,
                                       "mlir_threading": "disabled"},
        "artifacts": {name: _record(out / name) for name in ("source.ttir", "lowered.mlir", "stderr.log")},
        "remaining_contracts": ["declared shape/stride/effect/alias ABI and whole-launch grid",
                                "numeric-policy equivalence and work accounting",
                                "source-to-output provenance correspondence",
                                "functional tensor results and Merlin capture packaging"],
    }
    _write_json(out / "stage.json", report)
    return report
