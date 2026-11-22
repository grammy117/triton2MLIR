#!/usr/bin/env python3
"""Execute the mixed transformer model through the public Merlin interface."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from t2m.automatic_model import translate_launches
from t2m.bundle import _json, _record, verify_bundle
from t2m.consumer import _host_worker
from t2m.merlin_backend import MerlinHostBackend
from t2m.model_bundle import write_model_bundle


def run(out, *, tokens=16):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    capture = out / "capture"
    script = Path(__file__).with_name("capture_transformer_model.py")
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture),
        "--tokens", str(tokens)], capture_output=True, text=True, timeout=120, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(out)))
    (out / "capture.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise ValueError("transformer capture failed; inspect " + str(out / "capture.log"))
    launches = json.loads((capture / "launches.json").read_text())
    bindings = translate_launches(launches, out / "launch-translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    bound = out / "bound"
    write_model_bundle(capture / "parent", bindings, bound, work=out / "native",
        python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], rtol=3e-4, atol=3e-5)
    verify_bundle(bound)
    stage_compile_inputs(bound, out / "stage")
    product = compile(StagedProgram(out / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]),
        output=out / "compiled")
    product.verify()
    from merlin.compiler.compilation import Compilation

    # Invoke from the public artifact roster, without incidental build files.
    published = out / "published"
    for artifact in product.artifacts:
        target = published / artifact.path.relative_to(product.directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifact.path, target)
    shutil.copyfile(product.directory / "compilation.json", published / "compilation.json")
    product = Compilation.load(published)
    product.verify()
    observations = []
    library = _record(product.directory / "host/model.so")
    for ordinal, example in enumerate(sorted((capture / "references").glob("*.npz"))):
        invocation = out / f"invoke-{ordinal}"
        invocation.mkdir()
        with np.load(example, allow_pickle=False) as reference:
            arguments = invocation / "arguments.npz"
            np.savez(arguments, in0=reference["in0"])
            _host_worker("invoke", Path(os.environ["T2M_MERLIN_ROOT"]),
                product.directory / product.metadata["runtime_stage"], product.directory / "host",
                [str(arguments), str(invocation)], dict(os.environ, TMPDIR=str(invocation)),
                os.environ["T2M_MERLIN_PYTHON"])
            with np.load(invocation / "outputs.npz", allow_pickle=False) as actual:
                if set(actual.files) != {"out0", "out1"}:
                    raise ValueError("transformer result ABI lost a block observation")
                errors = []
                for name in ("out0", "out1"):
                    assert actual[name].shape == reference[name].shape and actual[name].dtype == reference[name].dtype
                    np.testing.assert_allclose(actual[name], reference[name], rtol=3e-4, atol=3e-5)
                    errors.append(float(np.max(np.abs(actual[name] - reference[name]), initial=0)))
            observations.append({"case": ordinal, "results": 2, "max_absolute_error": errors,
                "arguments": _record(arguments), "outputs": _record(invocation / "outputs.npz")})
    if _record(product.directory / "host/model.so") != library:
        raise ValueError("replay changed the compiled transformer artifact")
    report = {"schema": "t2m.two-transformer-demo.v1", "tokens": tokens,
        "architecture": "two-pre-norm-single-head-ReLU-transformer-blocks",
        "first_block": "ordinary-PyTorch-model2MLIR", "second_block": "automatically-discovered-Triton-source",
        "manual_kernel_bindings": False, "gpu_kernel_executed": False,
        "artifact_only_publication": True,
        "native_reference_comparison": "passed", "public_merlin_host_execution": "passed",
        "rtol": 3e-4, "atol": 3e-5, "invocations": observations,
        "launch_record": _record(capture / "launches.json"), "library": library,
        "accelerator_performance": "not-qualified"}
    _json(out / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, choices=(13, 16, 32), default=16)
    args = parser.parse_args()
    print(json.dumps(run(args.out, tokens=args.tokens), indent=2))
