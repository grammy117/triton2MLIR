"""Complete model/TTIR captures through execution-produced result descriptors."""

import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_data_size_model import implementation

from t2m.bundle import _record
from t2m.consumer import (
    check_merlin_host,
    check_merlin_interface,
    invoke_merlin_host,
    observe_merlin_host,
)
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle

REQUIRED = all(
    os.environ.get(name)
    for name in (
        "T2M_NATIVE_PYTHON",
        "T2M_MODEL2MLIR_PYTHON",
        "TRITON_SHARED_OPT",
        "T2M_MERLIN_ROOT",
        "T2M_MERLIN_PYTHON",
        "T2M_MERLIN_CLANG",
    )
)
pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing native/model2MLIR/Merlin tools required"
)


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    root = tmp_path_factory.mktemp("execution-result-parent")
    script = Path(__file__).resolve().parents[1] / "scripts/capture_data_size_model.py"
    done = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--out",
            str(root / "parent"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=dict(
            os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)
        ),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return root / "parent"


@pytest.fixture(scope="module", params=["tensor", "buffer", "guarded-buffer"])
def selected(tmp_path_factory, parent, request):
    root = tmp_path_factory.mktemp("execution-result-" + request.param)
    kernel = implementation(root, request.param)
    package = root / "bound"
    write_model_bundle(
        parent,
        [
            KernelBinding(
                "t2m_demo::data_size_prefix",
                kernel,
                runtime_operands={"n": "n", "capacity": "capacity"},
            )
        ],
        package,
        work=root / "native",
        python=os.environ["T2M_NATIVE_PYTHON"],
    )
    arguments = root / "native/arguments/arguments.npz"
    consumer = root / "consumer-owned"
    interface = check_merlin_interface(
        package,
        consumer,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        result_mode="execution-owned",
    )
    outputs, compilation = check_merlin_host(
        consumer,
        arguments,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"],
    )
    return package, consumer, interface, outputs, compilation


def test_complete_captured_model_preserves_every_original_pytorch_result(selected):
    package, consumer, interface, outputs, compilation = selected
    assert interface["schema"] == "t2m.merlin-interface-check.v3"
    assert interface["compiler_input_staging"] == "passed"
    assert interface["runtime_generation"].startswith("not-performed")
    signature = json.loads((consumer / "compiler/signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v6"
    assert len(signature["outputs"]) == len(outputs) == 7
    assert "shape_program" not in signature and "word_bindings" not in signature
    assert compilation["schema"] == "t2m.merlin-host-compilation.v2"
    assert compilation["whole_body_lowering"] == "passed"
    assert compilation["evaluator_data_supplied_to_compiler"] is False
    with np.load(package / "goldens.npz", allow_pickle=False) as saved:
        for i, out in enumerate(outputs):
            reference = saved[f"out{i}"]
            assert (out.dtype, out.shape, out.tobytes()) == (
                reference.dtype,
                reference.shape,
                reference.tobytes(),
            )
    assert not any(
        (consumer / "compiler" / name).exists()
        for name in ("inputs.npz", "golden.npy", "goldens.npz", "extra.npz")
    )


def test_one_compilation_handles_changed_shapes_counts_parameters_and_empty_results(
    selected, tmp_path
):
    _package, consumer, *_ = selected
    library = _record(consumer / "host/model.so")
    program = _record(consumer / "compiler/program.mlir")
    for capacity, n, weight in (
        (2, 0, 31),
        (3, 7, -11),
        (31, 31, 17),
        (13, 31, -27),
        (31, 1, 91),
        (13, 0, -19),
    ):
        x = np.arange(capacity, dtype=np.int32) * 3 - 7
        parameter = np.asarray(weight, np.int32)
        count = np.asarray(n, np.int64)
        original = x + parameter
        first = np.zeros(n, dtype=np.int32)
        first[: min(capacity, n)] = original[: min(capacity, n)]
        expected = (first, first, original, x, count, parameter, first[::2])
        arguments = tmp_path / f"arguments-{capacity}-{n}.npz"
        np.savez(arguments, in0=parameter, in1=x, in2=count)
        outputs, report = invoke_merlin_host(
            consumer,
            arguments,
            tmp_path / f"invoke-{capacity}-{n}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert report["schema"] == "t2m.merlin-host-evaluation.v2"
        assert report["functional_inputs_preserved"]
        assert len(report["result_observations"]) == report["invocations"] == 2
        for observation in report["result_observations"]:
            assert observation["source_calls"] == 1
            assert observation["shapes"] == [list(a.shape) for a in expected]
        for out, reference in zip(outputs, expected, strict=True):
            assert (out.dtype, out.shape, out.tobytes()) == (
                reference.dtype,
                reference.shape,
                reference.tobytes(),
            )
        assert _record(consumer / "host/model.so") == library
        assert _record(consumer / "compiler/program.mlir") == program


def test_invalid_counts_and_source_domains_abort_the_existing_library(
    selected, tmp_path
):
    _package, consumer, *_ = selected
    for capacity, n in ((13, -1), (13, 32), (1, 7), (32, 7)):
        arguments = tmp_path / f"invalid-{capacity}-{n}.npz"
        np.savez(
            arguments,
            in0=np.asarray(17, np.int32),
            in1=np.arange(capacity, dtype=np.int32),
            in2=np.asarray(n, np.int64),
        )
        destination = tmp_path / f"failed-{capacity}-{n}"
        outcome = observe_merlin_host(
            consumer,
            arguments,
            destination,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert outcome["returncode"] == -signal.SIGABRT
        assert not (destination / "outputs.npz").exists()


def test_rehashed_result_types_cannot_redirect_the_compiled_descriptor_roster(
    selected, tmp_path
):
    _package, consumer, *_ = selected
    copied = tmp_path / "consumer"
    shutil.copytree(consumer, copied)
    signature_path = copied / "compiler/signature.json"
    signature = json.loads(signature_path.read_text())
    signature["outputs"][0]["shape"] = [999]
    signature_path.write_text(json.dumps(signature, sort_keys=True, indent=2) + "\n")
    manifest_path = copied / "compiler/compile-inputs.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["members"]["signature.json"] = _record(signature_path)["sha256"]
    manifest_path.write_text(json.dumps(manifest))
    compiled_path = copied / "host/compilation.json"
    compiled = json.loads(compiled_path.read_text())
    compiled["source_signature"] = _record(signature_path)
    compiled_path.write_text(json.dumps(compiled))
    arguments = tmp_path / "arguments.npz"
    np.savez(
        arguments,
        in0=np.asarray(17, np.int32),
        in1=np.arange(13, dtype=np.int32),
        in2=np.asarray(7, np.int64),
    )
    with pytest.raises(ValueError, match="actual source"):
        invoke_merlin_host(
            copied,
            arguments,
            tmp_path / "failed",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
    assert not (tmp_path / "failed/outputs.npz").exists()
