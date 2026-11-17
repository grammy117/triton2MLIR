"""Runtime input words size real Triton results without compiler specialization."""

import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

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


def implementation(root, profile):
    from t2m.examples.data_size_kernels import data_size_prefix

    source = root / "source.ttir"
    source.write_text(
        capture_ttir(
            data_size_prefix,
            {
                "x": "*i32",
                "n": "i64",
                "capacity": "i64",
                "out": "*i32",
                "B": "constexpr",
            },
            {"B": 8},
        )
    )
    abi = TensorABI(
        "data_size_prefix",
        (
            TensorBuffer("x", ("capacity",), dtype="i32"),
            TensorBuffer("out", ("n",), "write", "i32"),
        ),
        grid=(GridExtent("n", 8),),
        runtime_scalars=(
            RuntimeScalar("n", "i64", 1, bounds=(0, 31)),
            RuntimeScalar("capacity", "i64", 2, bounds=(0, 31)),
        ),
    )
    return convert_shared(
        source,
        abi,
        root / "translation",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    root = tmp_path_factory.mktemp("data-size-capture")
    script = Path(__file__).resolve().parents[1] / "scripts/capture_data_size_model.py"
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
    root = tmp_path_factory.mktemp("data-size-" + request.param)
    kernel = implementation(root, request.param)
    write_model_bundle(
        captured,
        [
            KernelBinding(
                "t2m_demo::data_size_prefix",
                kernel,
                runtime_operands={"n": "n", "capacity": "capacity"},
            )
        ],
        root / "bound",
        work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_host=True,
        merlin_clang=os.environ["T2M_MERLIN_CLANG"],
    )
    return root / "bound", captured


def test_complete_package_declares_source_word_allocation_and_preserves_parent_bytes(
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
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert trace["model_shape_profile"] == "t2m.symbolic-model-shapes.v3"
    assert trace["argument_binding_profile"] == "t2m.checked-word-runtime-arguments.v1"
    assert len(trace["calls"]) == 2
    for call in trace["calls"]:
        assert call["runtime_operand_bindings"]["n"]["adapter"] == "checked-size-scalar"
        assert call["runtime_operand_bindings"]["n"]["integer_range"] == [0, 31]
    signature = json.loads((out / "consumer/compiler/signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v5"
    assert signature["shape_program"]["schema"] == "logical-size-program.v2"
    assert signature["word_bindings"] == {
        "w2": {"argument": 2, "indices": [], "dtype": "i64"}
    }
    assert len(signature["inputs"]) == 3 and len(signature["outputs"]) == 7
    for name in ("execution.json", "consumer/host/execution.json"):
        assert json.loads((out / name).read_text())["result_count"] == 7


def test_one_library_sizes_every_result_from_independent_capacity_and_count_words(
    tmp_path, packaged
):
    from t2m.consumer import invoke_merlin_host, observe_merlin_host

    out, _parent = packaged
    library, program = (
        _record(out / "consumer/host/model.so"),
        _record(out / "consumer/compiler/program.mlir"),
    )
    for ordinal, (capacity, n) in enumerate(
        ((2, 0), (2, 1), (13, 7), (5, 31), (31, 3), (31, 31), (7, 8))
    ):
        weight = np.asarray(17 if ordinal % 2 == 0 else -(1 << 31), np.int32)
        x, count = (
            np.arange(capacity, dtype=np.int32) * 53 - 19,
            np.asarray(n, np.int64),
        )
        original = x + weight
        expected = np.zeros(n, np.int32)
        expected[: min(n, capacity)] = original[: min(n, capacity)]
        references = [expected, expected, original, x, count, weight, expected[::2]]
        caller = tmp_path / f"inputs-{capacity}-{n}.npz"
        np.savez(caller, in0=weight, in1=x, in2=count)
        actual, report = invoke_merlin_host(
            out / "consumer",
            caller,
            tmp_path / f"invoke-{capacity}-{n}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert report["functional_inputs_preserved"] and len(actual) == 7
        for value, reference in zip(actual, references, strict=True):
            assert (value.dtype, value.shape, value.tobytes()) == (
                reference.dtype,
                reference.shape,
                reference.tobytes(),
            )
        assert _record(out / "consumer/host/model.so") == library
        assert _record(out / "consumer/compiler/program.mlir") == program
    for capacity, n in ((13, 32), (1, 7), (32, 7)):
        caller = tmp_path / f"invalid-{capacity}-{n}.npz"
        np.savez(
            caller,
            in0=np.asarray(17, np.int32),
            in1=np.arange(capacity, dtype=np.int32),
            in2=np.asarray(n, np.int64),
        )
        failed = tmp_path / f"failed-{capacity}-{n}"
        outcome = observe_merlin_host(
            out / "consumer",
            caller,
            failed,
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert outcome["returncode"] == -signal.SIGABRT
        assert not (failed / "outputs.npz").exists()
    caller = tmp_path / "negative.npz"
    np.savez(
        caller,
        in0=np.asarray(17, np.int32),
        in1=np.arange(13, dtype=np.int32),
        in2=np.asarray(-1, np.int64),
    )
    with pytest.raises(ValueError):
        invoke_merlin_host(
            out / "consumer",
            caller,
            tmp_path / "negative",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
    assert not (tmp_path / "negative/outputs.npz").exists()


def test_complete_data_sized_model_executes_on_spike(tmp_path, packaged):
    from scripts import qualify_spike_model as target

    out, _parent = packaged
    options = {
        "merlin_root": os.environ["T2M_MERLIN_ROOT"],
        "python": os.environ["T2M_MERLIN_PYTHON"],
    }
    report = target.qualify(out, tmp_path / "target", comparison="words", **options)
    assert target.verify(tmp_path / "target", **options) == report
    assert report["evaluation"]["result_count"] == 7


@pytest.mark.parametrize("damage", ("binding", "profile", "valid-program", "result"))
def test_rehashed_word_signature_cannot_replace_actual_storage_or_source_computation(
    tmp_path, packaged, damage
):
    from t2m.consumer import invoke_merlin_host

    out, _parent = packaged
    consumer = tmp_path / "consumer"
    shutil.copytree(out / "consumer", consumer)
    path = consumer / "compiler/signature.json"
    signature = json.loads(path.read_text())
    if damage == "binding":
        signature["word_bindings"]["w2"]["argument"] = 0
    elif damage == "profile":
        signature["schema"] = "merlin.compile-signature.v4"
    elif damage == "result":
        signature["outputs"][0]["shape"] = [1]
    else:
        from t2m.shape_expressions import Program

        program = Program(signature["shape_program"]["nodes"])
        signature["outputs"][0]["shape"] = [program.dimension(7)]
        roots = [
            size
            for row in signature["outputs"]
            for size in row["shape"]
            if isinstance(size, str) and size not in signature["dimension_bindings"]
        ]
        signature["shape_program"] = program.export(roots, word_profile=True)
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
        in1=np.arange(13, dtype=np.int32),
        in2=np.asarray(7, np.int64),
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
