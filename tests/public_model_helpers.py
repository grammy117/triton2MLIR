"""Exercise complete model packages through Merlin's actual public compiler."""

import json
import os
import shutil
import signal
from pathlib import Path

import numpy as np

from t2m.bundle import _record
from t2m.consumer import _host_worker


def compile_public(package, root, *, result_mode=None):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import (
        load_compile_inputs,
        stage_compile_inputs,
    )

    from t2m.merlin_backend import MerlinHostBackend

    stage_compile_inputs(package, root / "stage", result_mode=result_mode)
    source = load_compile_inputs(root / "stage")
    product = compile(StagedProgram(root / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]),
        output=root / "compiled")
    product.verify()
    assert source.source_profile == "derived-tensor-model"
    assert (product.directory / "frontend.mlir").read_text() == (product.directory / "compiler/program.mlir").read_text()
    assert product.metadata["runtime_stage"] == "compiler"
    assert not any(path.name in {"inputs.npz", "golden.npy", "goldens.npz", "extra.npz"}
                   for path in product.directory.rglob("*") if path.is_file())
    record = json.loads((product.directory / "compiler/source-bindings.json").read_text())
    assert not any(row["kind"] == "buffer" for row in record["initializers"])
    if source.signature["schema"] == "merlin.compile-signature.v6":
        from merlin.compiler.compilation import Compilation

        # A portable artifact bundle contains every member its runtime opens.
        # Copy only the public artifact roster, not incidental build products.
        destination = root / "published"
        for artifact in product.artifacts:
            target = destination / artifact.path.relative_to(product.directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact.path, target)
        shutil.copyfile(product.directory / "compilation.json", destination / "compilation.json")
        product = Compilation.load(destination)
        product.verify()
    return product, source, record


def invoke_public(product, arrays, expected, root):
    root.mkdir()
    arguments = root / "arguments.npz"
    np.savez(arguments, **{f"in{i}": value for i, value in enumerate(arrays)})
    library = _record(product.directory / "host/model.so")
    stage = product.directory / product.metadata["runtime_stage"]
    _host_worker("invoke", Path(os.environ["T2M_MERLIN_ROOT"]), stage, product.directory / "host",
        [str(arguments), str(root)], dict(os.environ, TMPDIR=str(root)), os.environ["T2M_MERLIN_PYTHON"])
    with np.load(root / "outputs.npz", allow_pickle=False) as results:
        assert len(results.files) == len(expected)
        for index, wanted in enumerate(expected):
            actual = results[f"out{index}"]
            assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
    assert _record(product.directory / "host/model.so") == library
    assert json.loads((root / "evaluation.json").read_text())["functional_inputs_preserved"]


def observe_public_shape_abort(product, arrays, root):
    from t2m.process_outcome import observe

    root.mkdir()
    arguments = root / "arguments.npz"
    np.savez(arguments, **{f"in{i}": value for i, value in enumerate(arrays)})
    worker = Path(__file__).resolve().parents[1] / "t2m/consumer_worker.py"
    command = [os.environ["T2M_MERLIN_PYTHON"], str(worker), "observe", os.environ["T2M_MERLIN_ROOT"],
        str(product.directory / product.metadata["runtime_stage"]), str(product.directory / "host"),
        str(arguments), str(root)]
    outcome = observe(command, root, environment=dict(os.environ, TMPDIR=str(root)), timeout=60)
    assert outcome["returncode"] == -signal.SIGABRT
    assert b"exported shape domain:" in (root / "stdout.bin").read_bytes() + (root / "stderr.bin").read_bytes()
    assert not (root / "outputs.npz").exists() and not (root / "evaluation.json").exists()
