"""Actual model2MLIR bindings preserve mapped reads and complete state tuples."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing model2MLIR, native and Merlin tools required")


@pytest.mark.parametrize("dtype", ("i64", "f32"))
def test_learned_model_maps_retain_weights_and_all_fourteen_results(tmp_path, dtype):
    from test_memory_maps import capture_state

    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY

    capture = tmp_path / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_memory_map_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture), "--dtype", dtype],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stdout + process.stderr
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    result = capture_state(kernel, dtype, True)
    names = [scalar.name for scalar in result.spec.runtime_scalars]
    bindings = [KernelBinding("t2m_demo::memory_map_state", result, tensor_views=POLICY,
        operands={"x": "x"}, scalar_arguments=dict(zip(names, values, strict=True)),
        runtime_arguments={name: name for name in names}) for values in ((13, 32, 3, 2), (31, 32, 4, -3), (0, 0, 4, 5))]
    out = tmp_path / "bound"
    write_model_bundle(capture, bindings, out, work=tmp_path / "qualification", python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"], merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_host=True, merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(out)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert len(trace["calls"]) == 3 and len(trace["kernels"]) == 1 and trace["model_abi"] == "unchanged"
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected:
        for path in (tmp_path / "qualification/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == [f"out{i}" for i in range(14)]
                for name in actual.files:
                    assert actual[name].shape == expected[name].shape and actual[name].dtype == expected[name].dtype
                    assert actual[name].tobytes() == expected[name].tobytes()
    if dtype == "i64":
        from t2m.cli import main

        configuration = {"schema": "t2m.kernel-bindings.v1", "kernels": [{
            "kernel": binding.qualname, "ttir": "kernel/source.ttir", "abi": result.spec.as_dict(),
            "operands": binding.operands, "tensor_views": POLICY, "scalar_arguments": binding.scalar_arguments,
            "runtime_arguments": binding.runtime_arguments, "memory_profile": "guarded-buffer", "address_policy": "source-words",
        } for binding in bindings]}
        path = tmp_path / "bindings.json"
        path.write_text(json.dumps(configuration))
        cli = tmp_path / "cli-bound"
        assert main(["bind-model", str(capture), "--bindings", str(path), "--out", str(cli),
            "--optimizer", os.environ["TRITON_SHARED_OPT"], "--python", os.environ["T2M_NATIVE_PYTHON"],
            "--merlin-root", os.environ["T2M_MERLIN_ROOT"], "--merlin-python", os.environ["T2M_MERLIN_PYTHON"],
            "--merlin-clang", os.environ["T2M_MERLIN_CLANG"], "--merlin-host"]) == 0
        verify_bundle(cli)
        assert (cli / "weights.safetensors").read_bytes() == (capture / "weights.safetensors").read_bytes()
        with (np.load(capture / "goldens.npz", allow_pickle=False) as expected,
              np.load(cli / "consumer/host/outputs.npz", allow_pickle=False) as actual):
            assert actual.files == [f"out{i}" for i in range(14)]
            for name in actual.files:
                assert actual[name].shape == expected[name].shape and actual[name].dtype == expected[name].dtype
                assert actual[name].tobytes() == expected[name].tobytes()
