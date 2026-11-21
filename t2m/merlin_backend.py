"""Public Merlin compilation adapter for imported TTIR kernel stages.

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
                or policy.get("numerical_transforms") != []
                or request.source.text != staged.program):
            raise ValueError("Merlin host import requires its unchanged source and non-transforming numeric policy")

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
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           MERLIN_COMPILE_TIMEOUT_S="30", TMPDIR=str(host))
        if self.compiler_python is not None:
            environment["MERLIN_COMPILER_PYTHON"] = self.compiler_python
        if self.clang is not None:
            environment["MERLIN_CLANG"] = self.clang
        _host_worker("compile", self.root, stage, host, [], environment, self.python)
        report = json.loads((host / "compilation.json").read_text())
        if (report.get("source_program") != _record(stage / "program.mlir")
                or report.get("source_signature") != _record(stage / "signature.json")
                or report.get("evaluator_data_supplied_to_compiler") is not False):
            raise ValueError("Merlin host compilation lost its exact compiler-only program/signature binding")
        request.preparation.revalidate(request.source)
        artifacts = tuple(Artifact.from_file(host / name, kind=kind, role="compiled-program")
            for name, kind in (("model.so", "shared-library"), ("model.ll", "llvm-ir"),
                               ("compilation.json", "compiler-report")))
        return Compilation(request.output, self.name, request.source.sha256, artifacts,
            metadata={"compiler_input_profile": request.preparation.stage.revalidate().source_profile,
                "source_policy": request.preparation.policy(), "execution_verified": False,
                "accelerator_execution": "not-performed"})
