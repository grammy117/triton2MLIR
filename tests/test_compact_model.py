"""Real scan/scatter compaction with execution-sized output and mixed profiles."""

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from t2m.bundle import _record, verify_bundle
from t2m.consumer import invoke_merlin_host
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
        "T2M_MERLIN_ROOT",
        "T2M_MERLIN_PYTHON",
        "T2M_MERLIN_CLANG",
        "TRITON_SHARED_OPT",
    )
)
pytestmark = pytest.mark.skipif(
    not REQUIRED, reason="existing source/model2MLIR/Merlin toolchains required"
)


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    root = tmp_path_factory.mktemp("compact-model-parent")
    script = Path(__file__).resolve().parents[1] / "scripts/capture_compact_model.py"
    done = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--out",
            str(root / "capture"),
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
    return root / "capture"


def pack_implementation(root):
    from t2m.examples.kernel_size_kernels import positive_pack

    root.mkdir(parents=True)
    source = root / "pack.ttir"
    source.write_text(
        capture_ttir(
            positive_pack,
            {
                "x": "*i32",
                "out": "*i32",
                "capacity": "i64",
                "n": "i64",
                "B": "constexpr",
            },
            {"B": 32},
        )
    )
    abi = TensorABI(
        "positive_pack",
        (
            TensorBuffer("x", ("capacity",), dtype="i32"),
            TensorBuffer("out", ("n",), "write", "i32"),
        ),
        runtime_scalars=(
            RuntimeScalar("capacity", "i64", 2, bounds=(0, 31)),
            RuntimeScalar("n", "i64", 3, bounds=(0, 31)),
        ),
    )
    return convert_shared(
        source,
        abi,
        root / "converted",
        optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="auto",
    )


@pytest.fixture(scope="module")
def selected(tmp_path_factory, parent):
    from test_kernel_size_binding import implementations

    root = tmp_path_factory.mktemp("compact-model")
    ordinary = implementations(root / "ordinary", "auto")
    pack = pack_implementation(root / "pack")
    bindings = [
        KernelBinding(
            "t2m_compact_demo::positive_count",
            ordinary[0].implementation,
            runtime_operands={"capacity": "capacity"},
        ),
        KernelBinding(
            "t2m_compact_demo::compact",
            pack,
            runtime_operands={"capacity": "capacity", "n": "n"},
        ),
        KernelBinding(
            "t2m_compact_demo::prefix",
            ordinary[1].implementation,
            runtime_operands={"capacity": "capacity", "n": "n"},
        ),
    ]
    package = write_model_bundle(
        parent,
        bindings,
        root / "package",
        work=root / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_clang=os.environ["T2M_MERLIN_CLANG"],
        merlin_host=True,
        result_mode="execution-owned",
    )
    return root, parent, pack, package


def test_real_compaction_keeps_parallel_scan_and_all_original_model_results(selected):
    _root, parent, pack, package = selected
    receipt = verify_bundle(package)
    assert receipt["schema"] == "t2m.bound-model-receipt.v2"
    stages = [op for op in pack.module.walk() if "prov.scan_distance" in op.attributes]
    assert len(stages) == 5 and all(op.name == "linalg.generic" for op in stages)
    selection = pack.trace["memory_selection"]
    assert [row["status"] for row in selection["attempts"]] == [
        "not-admitted",
        "not-admitted",
        "admitted",
    ]
    assert selection["attempts"][-1]["address_policy"] == "source-words"
    assert all(
        "integer range proof" in row["diagnostic"] for row in selection["attempts"][:-1]
    )
    assert len({op.attributes["prov.composite_id"].data for op in stages}) == 1
    assert any(op.name == "memref.store" for op in pack.module.walk())
    assert not any(
        op.name.startswith(
            ("tt.", "ttx.", "tts.", "ttg.", "gpu.", "nvgpu.", "nvvm.", "rocdl.")
        )
        for op in pack.module.walk()
    )
    compatibility = json.loads((package / "compatibility.json").read_text())
    profiles = [row["profile"] for row in compatibility["kernel_profiles"].values()]
    assert profiles.count("shared-functional-tensor") == 2
    assert profiles.count("shared-source-word-buffer") == 1
    for name, destination in receipt["parent_artifact_bindings"].items():
        assert (package / destination).read_bytes() == (parent / name).read_bytes()
    with (
        np.load(package / "consumer/host/outputs.npz", allow_pickle=False) as actual,
        np.load(parent / "goldens.npz", allow_pickle=False) as reference,
    ):
        for i in range(7):
            a, b = actual[f"out{i}"], reference[f"out{i}"]
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
        np.testing.assert_array_equal(
            actual["out0"], np.asarray([1, 6, 2, 5, 3, 4], np.int32)
        )
    trace = json.loads((package / "kernel-bindings.json").read_text())
    assert len(trace["calls"]) == 3 and "runtime_shape_contract" not in trace
    assert sorted(
        len(row["memory_selection"]["attempts"]) for row in trace["kernels"].values()
    ) == [1, 1, 3]
    assert (
        hashlib.sha256((package / "model.mlir").read_bytes()).hexdigest()
        == trace["mlir_sha256"]
    )


def test_one_published_library_preserves_sparse_masks_order_overflow_and_empty_results(
    selected, tmp_path
):
    _root, _parent, _pack, package = selected
    library = _record(package / "consumer/host/model.so")
    program = _record(package / "consumer/compiler/program.mlir")
    domains = [
        (np.asarray([0, -1], np.int32), 0),
        (np.asarray([4, -7, 2], np.int32), 0),
        (np.asarray([1, -1] * 15 + [7], np.int32), 0),
        (np.arange(31, dtype=np.int32), 100),
        (np.asarray([0, -1, 1, -(1 << 31), (1 << 31) - 1] * 6 + [7], np.int32), 1),
        (np.asarray([-9, 3, 8, -2, 0, 5, -1, 6, 1, -3, 7, 2, -8], np.int32), -3),
    ]
    for ordinal, (x, weight) in enumerate(domains):
        parameter = np.asarray(weight, np.int32)
        original = np.add(x, parameter, dtype=np.int32)
        first = original[original > 0].copy()
        count = np.asarray([len(first)], np.int64)
        expected = (first, first, original, x, count, parameter, first[::2])
        arguments = tmp_path / f"arguments-{ordinal}.npz"
        np.savez(arguments, in0=parameter, in1=x)
        actual, report = invoke_merlin_host(
            package / "consumer",
            arguments,
            tmp_path / f"invoke-{ordinal}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        assert report["functional_inputs_preserved"] and report["invocations"] == 2
        assert all(
            row["source_calls"] == 1
            and row["shapes"] == [list(value.shape) for value in expected]
            for row in report["result_observations"]
        )
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
        assert _record(package / "consumer/host/model.so") == library
        assert _record(package / "consumer/compiler/program.mlir") == program


def test_public_bind_cli_selects_each_kernel_representation_from_source_only(
    selected, tmp_path
):
    _root, parent, _pack, package = selected
    receipt = verify_bundle(package)
    trace = json.loads((package / "kernel-bindings.json").read_text())
    kernels = []
    for index, call in enumerate(trace["calls"]):
        directory = package / receipt["kernel_artifacts"][call["implementation_sha256"]]
        source = tmp_path / f"kernel-{index}.ttir"
        source.write_bytes((directory / "source.ttir").read_bytes())
        kernels.append(
            {
                "kernel": call["kernel"],
                "ttir": source.name,
                "abi": json.loads((directory / "spec.json").read_text()),
                "runtime_operands": call["runtime_operand_arguments"],
            }
        )
    configuration = tmp_path / "bindings.json"
    configuration.write_text(
        json.dumps({"schema": "t2m.kernel-bindings.v1", "kernels": kernels})
    )
    output = tmp_path / "cli-package"
    command = [
        sys.executable,
        "-m",
        "t2m",
        "bind-model",
        str(parent),
        "--bindings",
        str(configuration),
        "--out",
        str(output),
        "--optimizer",
        os.environ["TRITON_SHARED_OPT"],
        "--memory-profile",
        "auto",
        "--result-mode",
        "execution-owned",
        "--merlin-host",
        "--python",
        os.environ["T2M_NATIVE_PYTHON"],
        "--merlin-root",
        os.environ["T2M_MERLIN_ROOT"],
        "--merlin-python",
        os.environ["T2M_MERLIN_PYTHON"],
        "--merlin-clang",
        os.environ["T2M_MERLIN_CLANG"],
    ]
    done = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=150,
        check=False,
        env=dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            TMPDIR=str(tmp_path),
        ),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(done.stdout)["merlin_host_execution"] == "passed"
    verify_bundle(output)
    selected_trace = json.loads((output / "kernel-bindings.json").read_text())
    assert sorted(
        len(row["memory_selection"]["attempts"])
        for row in selected_trace["kernels"].values()
    ) == [1, 1, 3]


def test_standalone_pack_refuses_unwritten_results_without_changing_scan(selected, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(os.environ["T2M_MERLIN_ROOT"]) / "src"))
    from merlin.llvmlower.abi import ScalarArg
    from merlin.llvmlower.owned_result_program import save
    from merlin.llvmlower.owned_results import compile_owned_results

    from t2m.consumer_features import select_features

    _root, _parent, pack, _package = selected
    program = compile_owned_results(
        pack.mlir_text, workdir=tmp_path / "program", entry="positive_pack",
        features=select_features(pack.module),
    )
    saved = save(program)
    x = np.asarray([4, -7, 2], np.int32)
    for n in (0, 1, 2):
        (actual,), report = program.execute([x, ScalarArg(3, "i64"), ScalarArg(n, "i64")])
        np.testing.assert_array_equal(actual, np.asarray([4, 2], np.int32)[:n])
        assert report["source_calls"] == 1
        np.savez(tmp_path / f"valid-{n}.npz", actual=actual, reference=x[x > 0][:n])
        (tmp_path / f"valid-{n}.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    worker = tmp_path / "invalid.py"
    worker.write_text('''import ctypes
import resource
import sys
import numpy as np
from merlin.llvmlower.abi import ScalarArg
from merlin.llvmlower.owned_result_program import load
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
libc = ctypes.CDLL(None)
libc.setbuf.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libc.setbuf(ctypes.c_void_p.in_dll(libc, "stdout"), None)
program = load(sys.argv[1])
outputs, _ = program.execute([
    np.asarray([4, -7, 2], np.int32), ScalarArg(3, "i64"), ScalarArg(3, "i64")
])
np.savez(sys.argv[2], result=outputs[0])
''')
    output = tmp_path / "invalid-output.npz"
    done = subprocess.run(
        [sys.executable, str(worker), str(saved), str(output)],
        capture_output=True, text=True, timeout=30, check=False,
        env=dict(os.environ, TMPDIR=str(tmp_path),
                 PYTHONPATH=str(Path(os.environ["T2M_MERLIN_ROOT"]) / "src")),
    )
    (tmp_path / "invalid.json").write_text(json.dumps({
        "returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr,
        "published_output": output.exists(),
    }, sort_keys=True) + "\n")
    assert done.returncode == -signal.SIGABRT, done.stdout + done.stderr
    assert "fresh output has an unwritten logical element: out" in done.stdout + done.stderr
    assert not output.exists()
