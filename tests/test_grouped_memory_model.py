"""Complete models retain packed read regions, learned state and all snapshots."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_grouped_memory import capture as translate, arguments
from test_spike_model import Q, TOOLS as TARGET_TOOLS

ROOT = Path(__file__).resolve().parents[1]
TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


@pytest.mark.parametrize("warps", (2, 4))
@pytest.mark.parametrize("selected", (False, True))
def test_model_reference_retains_typed_source_words_and_program_local_reads(warps, selected):
    spec = importlib.util.spec_from_file_location("independent_group_memory_reference", ROOT / "t2m/examples/grouped_memory_reference.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    for dtype in ("i32", "i64", "f32"):
        for n, capacity, iterations, delta in ((0, 0, 3, 2), (1, 1, 0, -3), (300, 320, 3, 2), (531, 550, 2, 5)):
            values, expected = arguments(n, capacity, iterations, delta, dtype, warps, selected)
            actual = oracle.reference(*values[:2], n, capacity, iterations, delta, warps=warps, selected=selected)
            for value, gold in zip(actual, expected, strict=True):
                assert (value.shape, value.dtype, value.tobytes()) == (gold.shape, gold.dtype, gold.tobytes())


@pytest.mark.skipif(not TOOLS, reason="existing model2MLIR, native and Merlin tools required")
@pytest.mark.parametrize("dtype,warps,while_loop", [("i64", 2, True), ("f32", 4, False)])
def test_learned_grouped_memory_model_retains_all_fifteen_results(tmp_path, dtype, warps, while_loop):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_source_layout import PROFILE
    from t2m.model_tensor_views import POLICY

    source = tmp_path / "source"
    script = ROOT / "scripts/capture_grouped_memory_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(source), "--dtype", dtype,
        "--warps", str(warps), *([] if while_loop else ["--for-loop"])], capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stdout + process.stderr
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    result = translate(kernel, dtype, warps, while_loop, True)
    mapping = {"target": "source_target", "num_warps": "source_warps", "threads_per_warp": "source_threads", "num_ctas": "source_ctas"}
    profile = {"source_target": "cuda:80", "source_warps": warps, "source_threads": 32,
        "source_ctas": 1, "while_loop": while_loop, "selected": True}
    names = ("n", "capacity", "iterations", "delta")
    bindings = [KernelBinding("t2m_demo::grouped_read_state", result, tensor_views=POLICY,
        scalar_arguments={**dict(zip(names, values, strict=True)), **profile},
        runtime_arguments={name: name for name in names}, source_layout_arguments=mapping)
        for values in ((300, 320, 3, 2), (319, 320, 1, -3), (0, 0, 2, 5))]
    out = tmp_path / "bound"
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    write_model_bundle(source, bindings, out, work=tmp_path / "host", python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=root, merlin_python=python, merlin_host=True, merlin_clang=clang)
    verify_bundle(out)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (source / name).read_bytes()
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert trace["source_layout_binding_profile"] == PROFILE
    assert len(trace["calls"]) == 3 and len(trace["kernels"]) == 1 and trace["model_abi"] == "unchanged"
    assert all(set(call["source_layout_bindings"]) == set(mapping) for call in trace["calls"])
    assert all(len(call["source_result_order"]) == 5 for call in trace["calls"])
    assert ("scf.while" if while_loop else "scf.for") in (out / "model.mlir").read_text()
    with np.load(source / "goldens.npz", allow_pickle=False) as expected:
        assert [expected[f"out{i}"].size for i in range(10, 15)] == [0] * 5
        for path in (tmp_path / "host/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == [f"out{i}" for i in range(15)]
                for key in actual.files:
                    assert (actual[key].shape, actual[key].dtype, actual[key].tobytes()) == (
                        expected[key].shape, expected[key].dtype, expected[key].tobytes())
    if TARGET_TOOLS:
        report = Q.qualify(out, tmp_path / "target", merlin_root=root, python=python, comparison="words")
        assert report["evaluation"]["result_count"] == 15
        assert Q.verify(tmp_path / "target", merlin_root=root, python=python) == report
