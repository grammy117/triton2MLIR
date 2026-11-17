"""Real dynamic models keep complete packages and reusable Merlin interfaces."""

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

REQUIRED = all(os.environ.get(name) for name in ("T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON",
    "TRITON_SHARED_OPT", "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing native/model2MLIR/Merlin environments required")


@pytest.fixture(scope="module", params=[False, True], ids=["copy", "weighted"])
def captured(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("symbolic-package-capture")
    parent = root / "parent"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_symbolic_copy_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(parent),
        *(["--weighted"] if request.param else [])], capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert process.returncode == 0, process.stdout + process.stderr
    return parent, request.param


@pytest.fixture(scope="module", params=["tensor", "buffer", "guarded-buffer"])
def packaged(tmp_path_factory, captured, request):
    from t2m.examples.live_scalar_kernels import live_extent_copy, live_extent_weighted

    parent, weighted = captured
    root = tmp_path_factory.mktemp("symbolic-package-" + request.param)
    kernel = live_extent_weighted if weighted else live_extent_copy
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, {"x": "*i32", "out": "*i32", "n": "i32", "B": "constexpr",
                                          **({"weight": "*i32"} if weighted else {})}, {"B": 8}))
    buffers = (TensorBuffer("x", ("n",), dtype="i32"),
               *((TensorBuffer("weight", (1,), dtype="i32"),) if weighted else ()),
               TensorBuffer("out", ("n",), "write", "i32"))
    spec = TensorABI(kernel.__name__, buffers, grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2 if weighted else 1, bounds=(0, 31)),))
    implementation = convert_shared(source, spec, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=request.param, address_policy="source-words" if request.param == "guarded-buffer" else "structured")
    out = root / "bound"
    name = "t2m_demo::symbolic_weighted" if weighted else "t2m_demo::symbolic_copy"
    write_model_bundle(parent, [KernelBinding(name, implementation, runtime_operands={"n": "n"})], out,
        work=root / "work", python=os.environ["T2M_NATIVE_PYTHON"], merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_host=True, merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    return out, parent, weighted


def test_dynamic_package_keeps_parent_data_and_independent_merlin_shape_signature(packaged):
    out, parent, weighted = packaged
    verify_bundle(out)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "input_order.json", "inputs.npz",
                 "extra.npz", "golden.npy", "goldens.npz"):
        assert (out / name).read_bytes() == (parent / name).read_bytes()
    signature = json.loads((out / "consumer/compiler/signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v3"
    name = "d1_0" if weighted else "d0_0"
    assert signature["dimension_bindings"] == {name: {"argument": 1 if weighted else 0, "axis": 0}}
    assert signature["inputs"][1 if weighted else 0]["shape"] == [name]
    assert [row["shape"] for row in signature["outputs"]] == [[name]] * 3 + ([[1]] if weighted else [])
    for file in ("execution.json", "consumer/host/execution.json"):
        assert json.loads((out / file).read_text())["result_count"] == (4 if weighted else 3)
    manifest = json.loads((out / "consumer/compiler/compile-inputs.json").read_text())
    assert not set(manifest["members"]) & {"inputs.npz", "golden.npy", "goldens.npz", "extra.npz", "execution.json"}


def test_one_model_library_handles_independent_shapes_controls_and_actual_guard_abort(tmp_path, packaged):
    from t2m.consumer import invoke_merlin_host, observe_merlin_host

    out, _parent, weighted = packaged
    library, program = _record(out / "consumer/host/model.so"), _record(out / "consumer/compiler/program.mlir")
    for index, length in enumerate((0, 1, 7, 31)):
        x = np.arange(length, dtype=np.int32) * 83 - 49
        weight = np.array([[17, -1, 0, -(1 << 31)][index]], dtype=np.int32)
        args = [weight, x] if weighted else [x]
        reference = [x, x ^ weight[0], x, weight] if weighted else [x, x, x]
        path = tmp_path / f"caller-{length}.npz"
        np.savez(path, **{f"in{i}": value for i, value in enumerate(args)})
        actual, report = invoke_merlin_host(out / "consumer", path, tmp_path / f"invoke-{length}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for value, expected in zip(actual, reference, strict=True):
            assert (value.shape, value.dtype, value.tobytes()) == (expected.shape, expected.dtype, expected.tobytes())
        assert _record(out / "consumer/host/model.so") == library
        assert _record(out / "consumer/compiler/program.mlir") == program
    path = tmp_path / "invalid.npz"
    args = [np.array([17], np.int32), np.arange(32, dtype=np.int32)] if weighted else [np.arange(32, dtype=np.int32)]
    np.savez(path, **{f"in{i}": value for i, value in enumerate(args)})
    failed = tmp_path / "invalid"
    outcome = observe_merlin_host(out / "consumer", path, failed,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert outcome["returncode"] == -signal.SIGABRT
    assert b"exported shape domain:" in (failed / "stdout.bin").read_bytes() + (failed / "stderr.bin").read_bytes()
    assert not (failed / "outputs.npz").exists() and not (failed / "evaluation.json").exists()


def test_complete_dynamic_model_runs_through_original_spike_runtime(tmp_path, packaged):
    from scripts import qualify_spike_model as target

    out, _parent, weighted = packaged
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    report = target.qualify(out, tmp_path / "target", comparison="words", **options)
    assert target.verify(tmp_path / "target", **options) == report
    assert report["evaluation"]["result_count"] == (4 if weighted else 3)


@pytest.mark.parametrize("rehash_compilation", [False, True])
def test_rehashed_signature_cannot_underallocate_a_previously_compiled_model(tmp_path, packaged, rehash_compilation):
    from t2m.consumer import invoke_merlin_host

    out, _parent, weighted = packaged
    changed = tmp_path / "consumer"
    shutil.copytree(out / "consumer", changed)
    signature = json.loads((changed / "compiler/signature.json").read_text())
    signature["outputs"][0]["shape"] = [1]
    (changed / "compiler/signature.json").write_text(json.dumps(signature))
    manifest = json.loads((changed / "compiler/compile-inputs.json").read_text())
    manifest["members"]["signature.json"] = _record(changed / "compiler/signature.json")["sha256"]
    (changed / "compiler/compile-inputs.json").write_text(json.dumps(manifest))
    if rehash_compilation:
        compilation = json.loads((changed / "host/compilation.json").read_text())
        compilation["source_signature"] = _record(changed / "compiler/signature.json")
        (changed / "host/compilation.json").write_text(json.dumps(compilation))
    args = [np.array([17], np.int32), np.arange(7, dtype=np.int32)] if weighted else [np.arange(7, dtype=np.int32)]
    arguments = tmp_path / "inputs.npz"
    np.savez(arguments, **{f"in{i}": value for i, value in enumerate(args)})
    with pytest.raises(ValueError, match="signature differs|executable source dimensions"):
        invoke_merlin_host(changed, arguments, tmp_path / "failed", merlin_root=os.environ["T2M_MERLIN_ROOT"],
                          python=os.environ["T2M_MERLIN_PYTHON"])
    assert not (tmp_path / "failed/outputs.npz").exists()
