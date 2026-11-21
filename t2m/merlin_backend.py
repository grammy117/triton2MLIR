"""Public Merlin compilation adapter for imported TTIR kernel/model stages.

The adapter selects the same existing whole-body Merlin native lowering used by
standalone qualification. It supplies no new arithmetic, source-language decoder
or numerical verdict. Runtime arguments and references are separate invocation
inputs and never enter this compiler call.
"""

import json
import os
import sys
from pathlib import Path

from .bundle import _record
from .consumer import _host_worker


class MerlinHostBackend:
    name = "triton2mlir.merlin-host"

    def __init__(self, *, merlin_root, python=None, compiler_python=None, clang=None):
        self.root = Path(merlin_root).resolve()
        self.python = str(python or sys.executable)
        self.compiler_python = str(compiler_python) if compiler_python is not None else None
        self.clang = str(clang) if clang is not None else None

    def validate_preparation(self, request):
        from merlin.compiler.frontend.intake.staged_projection import POLICY, PROFILES

        preparation = request.preparation
        if preparation is None:
            raise ValueError("Merlin host import requires an explicit StagedProgram preparation")
        preparation.revalidate(request.source)
        staged = preparation.stage.revalidate()
        policy = preparation.policy()
        if (staged.source_profile not in PROFILES or policy.get("schema") != POLICY
                or policy.get("source_profile") != staged.source_profile
                or policy.get("source_sha256") != staged.original_source_sha256
                or policy.get("operation_semantics") != "retain-imported-operations"
                or policy.get("numerical_transforms") != []):
            raise ValueError("Merlin host import requires its source-bound preparation and non-transforming numeric policy")

    def validate(self, request):
        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

        if request.preparation is None or request.source.kind != "mlir":
            raise ValueError("Merlin host import requires a prepared standard MLIR program")
        if request.options.parameters or any(value is not None for value in (
                request.options.system, request.options.platform, request.options.harts)):
            raise ValueError("Merlin host import accepts no unimplemented compiler or deployment options")
        parse_mlir_text(request.source.text).verify()

    def compile(self, request):
        from merlin.compiler.compilation import Artifact, Compilation

        self.validate_preparation(request)
        self.validate(request)
        if request.output.exists():
            raise FileExistsError(request.output)
        request.output.mkdir()
        host = request.output / "host"
        host.mkdir()
        stage = request.preparation.stage.directory
        staged = request.preparation.stage.revalidate()
        execution_members = []
        if staged.source_profile == "derived-tensor-model":
            from merlin.compiler.frontend.bindings import imported_runtime_signature

            signature = imported_runtime_signature(staged, request.preparation.bound)
            original = json.loads((stage / "compile-inputs.json").read_bytes())
            stage = request.output / "compiler"
            stage.mkdir()
            values = {"program.mlir": request.source.text.encode(),
                "signature.json": (json.dumps(signature, indent=2, sort_keys=True) + "\n").encode(),
                "source-bindings.json": (json.dumps(request.preparation.bound.record, indent=2, sort_keys=True) + "\n").encode(),
                "precision-policy.json": staged.precision_policy_document}
            for name, raw in values.items():
                (stage / name).write_bytes(raw)
            manifest = {"schema": "merlin.bound-compile-inputs.v1",
                "preparation": request.preparation.record(),
                "members": {name: _record(stage / name)["sha256"] for name in values},
                **{key: original[key] for key in ("result_mode", "result_profile") if key in original}}
            (stage / "compile-inputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            execution_members = [Artifact.from_file(stage / name, kind="prepared-compiler-input", role="compiler-input")
                                 for name in (*values, "compile-inputs.json")]
        identities = {name: _record(stage / name) for name in
                      ("program.mlir", "signature.json", "compile-inputs.json")}
        if execution_members:
            identities.update({name: _record(stage / name) for name in values})
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           MERLIN_COMPILE_TIMEOUT_S="30", TMPDIR=str(host))
        if self.compiler_python is not None:
            environment["MERLIN_COMPILER_PYTHON"] = self.compiler_python
        if self.clang is not None:
            environment["MERLIN_CLANG"] = self.clang
        _host_worker("compile", self.root, stage, host, [], environment, self.python)
        report = json.loads((host / "compilation.json").read_text())
        if (report.get("source_program") != identities["program.mlir"]
                or report.get("source_signature") != identities["signature.json"]
                or any(_record(stage / name) != pin for name, pin in identities.items())
                or report.get("evaluator_data_supplied_to_compiler") is not False):
            raise ValueError("Merlin host compilation lost its exact compiler-only program/signature binding")
        request.preparation.revalidate(request.source)
        artifacts = tuple(Artifact.from_file(host / name, kind=kind, role="compiled-program")
            for name, kind in (("model.so", "shared-library"), ("model.ll", "llvm-ir"),
                               ("compilation.json", "compiler-report"))) + tuple(execution_members)
        if staged.signature["schema"] == "merlin.compile-signature.v6":
            from merlin.compiler.lower.options.owned_result_program import (
                FILENAME,
                load,
            )

            owned = load(host / FILENAME)
            owned.verify()
            record = json.loads((host / FILENAME).read_bytes())
            artifacts += tuple(Artifact.from_file(host / name, kind="owned-result-runtime-input", role="runtime")
                for name in (FILENAME, *record["artifacts"]) if name not in {"model.so", "model.ll"})
        return Compilation(request.output, self.name, request.source.sha256, artifacts,
            metadata={"compiler_input_profile": request.preparation.stage.revalidate().source_profile,
                "source_policy": request.preparation.policy(), "execution_verified": False,
                "runtime_stage": "compiler" if execution_members else "frontend",
                "accelerator_execution": "not-performed"})
