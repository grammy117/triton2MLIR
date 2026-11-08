"""Optional strict native MLIR verification, isolated from the frontend's dependencies."""

import json
import hashlib
import os
import subprocess
import sys


_SCRIPT = '''
import json, sys
from importlib.metadata import version
from torch_mlir import ir, passmanager
with ir.Context() as context:
    context.allow_unregistered_dialects = False
    module = ir.Module.parse(sys.stdin.read())
    if not module.operation.verify():
        raise ValueError("native MLIR verification failed")
    lower = "--lower-to-llvm" in sys.argv
    if lower:
        pipelines = (
            "builtin.module(scf-forall-to-for)",
            "builtin.module(canonicalize,cse,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map})",
            "builtin.module(func.func(convert-linalg-to-loops),lower-affine,convert-scf-to-cf,expand-strided-metadata,convert-math-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)",
        )
        for pipeline in pipelines:
            passmanager.PassManager.parse(pipeline).run(module.operation)
        def check(operation):
            if operation.name.split(".")[0] not in ("builtin", "llvm") or operation.name == "builtin.unrealized_conversion_cast":
                raise ValueError("LLVM legalization left " + operation.name)
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.operations:
                        check(child.operation)
        check(module.operation)
        if not module.operation.verify():
            raise ValueError("lowered LLVM module did not verify")
print(json.dumps({"native_mlir_verified": True, "provider": "torch-mlir",
                  "provider_version": version("torch-mlir"),
                  "unregistered_dialects_allowed": False,
                  "llvm_lowering_verified": lower,
                  "execution": "not-performed"}))
'''


def verify_mlir(mlir_text, *, python=None, lower_to_llvm=False):
    """Use existing native bindings; never install Torch or build LLVM implicitly.

    python may identify another user-owned environment with torch-mlir installed.
    This validates registered upstream operations, not just a permissive parser.
    """
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        process = subprocess.run([str(python or sys.executable), "-c", _SCRIPT,
                                  *(["--lower-to-llvm"] if lower_to_llvm else [])],
                                 input=mlir_text, text=True, capture_output=True,
                                 timeout=30, env=environment)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("native MLIR verification exceeded its 30-second limit") from exc
    if process.returncode:
        raise ValueError("strict native MLIR verification failed (existing torch-mlir bindings required): "
                         + process.stderr[-4000:])
    result = json.loads(process.stdout)
    result["input_sha256"] = hashlib.sha256(mlir_text.encode()).hexdigest()
    return result
