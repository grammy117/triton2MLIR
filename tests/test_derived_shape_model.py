"""Real derived extents survive complete package, reusable host and target calls."""

import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
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
def captured(tmp_path_factory):
    root = tmp_path_factory.mktemp("derived-shape-capture")
    script = (
        Path(__file__).resolve().parents[1] / "scripts/capture_derived_shape_model.py"
    )
    process = subprocess.run(
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
    assert process.returncode == 0, process.stdout + process.stderr
    return root / "parent"


@pytest.fixture(scope="module", params=["tensor", "buffer", "guarded-buffer"])
def packaged(tmp_path_factory, captured, request):
    from test_live_scalar_model import implementation

    root = tmp_path_factory.mktemp("derived-shape-" + request.param)
    kernel = implementation(root, "i32", extent=True, profile=request.param)
    write_model_bundle(
        captured,
        [KernelBinding("t2m_demo::derived_copy", kernel, runtime_operands={"n": "n"})],
        root / "bound",
        work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_host=True,
        merlin_clang=os.environ["T2M_MERLIN_CLANG"],
    )
    return root / "bound", captured


def test_derived_package_preserves_source_archives_and_independent_shape_program(
    packaged,
):
    out, parent = packaged
    verify_bundle(out)
    for name in (
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "extra.npz",
        "input_order.json",
        "golden.npy",
        "goldens.npz",
    ):
        assert (out / name).read_bytes() == (parent / name).read_bytes()
    signature = json.loads((out / "consumer/compiler/signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v4"
    assert signature["model_shape_profile"] == "t2m.symbolic-model-shapes.v2"
    assert signature["shape_program"]["schema"] == "logical-size-program.v1"
    assert signature["dimension_bindings"] == {
        f"d{i}_0": {"argument": i, "axis": 0} for i in (1, 2)
    }
    assert len(signature["inputs"]) == 3 and len(signature["outputs"]) == 7
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert trace["runtime_shape_contract"]["schema"] == "t2m.runtime-shapes.v3"
    assert len(trace["calls"]) == 2
    for call in trace["calls"]:
        assert call["runtime_operand_bindings"]["n"]["unsigned_integer_range"] == [
            2,
            15,
        ]
    manifest = json.loads((out / "consumer/compiler/compile-inputs.json").read_text())
    assert not set(manifest["members"]) & {
        "inputs.npz",
        "golden.npy",
        "goldens.npz",
        "extra.npz",
        "execution.json",
    }


def test_one_library_retains_every_derived_result_across_lengths_and_weights(
    tmp_path, packaged
):
    from t2m.consumer import invoke_merlin_host, observe_merlin_host

    out, _parent = packaged
    library, program = (
        _record(out / "consumer/host/model.so"),
        _record(out / "consumer/compiler/program.mlir"),
    )
    for ordinal, n in enumerate((4, 5, 7, 31)):
        x = np.arange(n, dtype=np.int32) * 7 - 19
        y = np.arange(n, dtype=np.int32) * -3
        weight = np.asarray([17, -1, 0, -(1 << 31)][ordinal], dtype=np.int32)
        original = x + y + weight
        trimmed = original[1::2]
        expected = [trimmed, trimmed, original, trimmed, x, y, weight]
        source = tmp_path / f"inputs-{n}.npz"
        np.savez(source, in0=weight, in1=x, in2=y)
        actual, report = invoke_merlin_host(
            out / "consumer",
            source,
            tmp_path / f"invoke-{n}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert report["functional_inputs_preserved"]
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected, strict=True):
            assert (result.dtype, result.shape, result.tobytes()) == (
                reference.dtype,
                reference.shape,
                reference.tobytes(),
            )
        assert _record(out / "consumer/host/model.so") == library
        assert _record(out / "consumer/compiler/program.mlir") == program
    for nx, ny in ((3, 3), (32, 32), (5, 6)):
        source = tmp_path / f"invalid-{nx}-{ny}.npz"
        np.savez(
            source,
            in0=np.asarray(17, np.int32),
            in1=np.arange(nx, dtype=np.int32),
            in2=np.arange(ny, dtype=np.int32),
        )
        failed = tmp_path / f"failed-{nx}-{ny}"
        outcome = observe_merlin_host(
            out / "consumer",
            source,
            failed,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert outcome["returncode"] == -signal.SIGABRT
        assert (
            b"exported shape domain:"
            in (failed / "stdout.bin").read_bytes()
            + (failed / "stderr.bin").read_bytes()
        )
        assert not (failed / "outputs.npz").exists()


def test_complete_derived_model_executes_on_spike(tmp_path, packaged):
    from scripts import qualify_spike_model as target

    out, _parent = packaged
    options = {
        "merlin_root": os.environ["T2M_MERLIN_ROOT"],
        "python": os.environ["T2M_MERLIN_PYTHON"],
    }
    report = target.qualify(out, tmp_path / "target", comparison="words", **options)
    assert target.verify(tmp_path / "target", **options) == report
    assert report["evaluation"]["result_count"] == 7


@pytest.mark.parametrize("damage", ("output", "program", "profile", "valid-program"))
def test_rehashed_signature_cannot_invent_derived_output_storage(
    tmp_path, packaged, damage
):
    from t2m.consumer import invoke_merlin_host

    out, _parent = packaged
    consumer = tmp_path / "consumer"
    shutil.copytree(out / "consumer", consumer)
    path = consumer / "compiler/signature.json"
    signature = json.loads(path.read_text())
    if damage == "output":
        signature["outputs"][0]["shape"] = [1]
    elif damage == "program":
        next(
            row
            for row in signature["shape_program"]["nodes"].values()
            if row["op"] == "const"
        )["value"] ^= 1
    elif damage == "valid-program":
        from t2m.shape_expressions import Program
        program = Program(signature["shape_program"]["nodes"])
        one = program.dimension(1)
        signature["outputs"][0]["shape"] = [one]
        roots = [size for row in signature["outputs"] for size in row["shape"]
                 if isinstance(size, str) and size not in signature["dimension_bindings"]]
        signature["shape_program"] = program.export(roots)
    else:
        signature["schema"] = "merlin.compile-signature.v3"
    path.write_text(json.dumps(signature))
    manifest = json.loads((consumer / "compiler/compile-inputs.json").read_text())
    manifest["members"]["signature.json"] = _record(path)["sha256"]
    (consumer / "compiler/compile-inputs.json").write_text(json.dumps(manifest))
    compilation = json.loads((consumer / "host/compilation.json").read_text())
    compilation["source_signature"] = _record(path)
    (consumer / "host/compilation.json").write_text(json.dumps(compilation))
    caller = tmp_path / "inputs.npz"
    np.savez(
        caller,
        in0=np.asarray(17, np.int32),
        in1=np.arange(7, dtype=np.int32),
        in2=np.arange(7, dtype=np.int32),
    )
    with pytest.raises(ValueError):
        invoke_merlin_host(
            consumer,
            caller,
            tmp_path / "refused",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
    assert not (tmp_path / "refused/outputs.npz").exists()
