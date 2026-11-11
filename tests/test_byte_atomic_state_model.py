"""Actual model2MLIR packages retain byte atomics, weights and every result."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

REQUIRED = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing model2MLIR/native/Merlin environments required")


@pytest.mark.parametrize("width,cas,typed_root", ((32, False, False), (64, False, True), (16, True, True), (32, True, False)))
def test_real_model_keeps_selected_atomic_recurrence_weights_and_seven_results(tmp_path, width, cas, typed_root):
    from t2m.bundle import verify_bundle
    from t2m.byte_atomic_state import convert
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY

    capture = tmp_path / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_byte_atomic_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture),
        "--width", str(width), *(["--cas"] if cas else []), *(["--typed-root"] if typed_root else [])],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stdout + process.stderr
    result = convert(tmp_path / "source", width=width, cas=cas, while_=True, carry=True,
                     root_dtype="i64" if typed_root else "i8", optimizer=os.environ["TRITON_SHARED_OPT"])
    names = [scalar.name for scalar in result.spec.runtime_scalars]
    bindings = [KernelBinding("t2m_demo::byte_atomic_state", result, tensor_views=POLICY,
        operands={"a": "a", "b": "b"}, scalar_arguments=dict(zip(names, values, strict=True)),
        runtime_arguments={name: name for name in names}) for values in ((3, 3, 64, True), (4, 2, 64, False), (0, 0, 0, False))]
    out = tmp_path / "bound"
    write_model_bundle(capture, bindings, out, work=tmp_path / "qualification", python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"], merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_host=True, merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(out)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert len(trace["calls"]) == 3 and len(trace["kernels"]) == 1 and trace["model_abi"] == "unchanged"
    host = json.loads((out / "consumer/host/execution.json").read_text())
    assert host["whole_body_lowering"] == host["compiler_input_staging"] == host["reference_comparison"] == "passed"
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected:
        for path in (tmp_path / "qualification/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == [f"out{i}" for i in range(7)]
                for name in actual.files:
                    assert actual[name].shape == expected[name].shape and actual[name].dtype == expected[name].dtype
                    assert actual[name].tobytes() == expected[name].tobytes()
    if width == 16 and cas and typed_root:
        from t2m.cli import main

        config = {"schema": "t2m.kernel-bindings.v1", "kernels": [{
            "kernel": binding.qualname, "ttir": "source/source.ttir", "abi": result.spec.as_dict(),
            "operands": binding.operands, "tensor_views": POLICY, "scalar_arguments": binding.scalar_arguments,
            "runtime_arguments": binding.runtime_arguments, "memory_profile": "guarded-buffer",
            "address_policy": "source-words", "effect_profile": "typed-atomics", "byte_order": "little",
        } for binding in bindings]}
        path = tmp_path / "bindings.json"
        path.write_text(json.dumps(config))
        cli_out = tmp_path / "cli-bound"
        assert main(["bind-model", str(capture), "--bindings", str(path), "--out", str(cli_out),
            "--optimizer", os.environ["TRITON_SHARED_OPT"], "--python", os.environ["T2M_NATIVE_PYTHON"],
            "--merlin-root", os.environ["T2M_MERLIN_ROOT"], "--merlin-python", os.environ["T2M_MERLIN_PYTHON"],
            "--merlin-clang", os.environ["T2M_MERLIN_CLANG"], "--merlin-host"]) == 0
        verify_bundle(cli_out)
        assert (cli_out / "weights.safetensors").read_bytes() == (capture / "weights.safetensors").read_bytes()
        with (np.load(capture / "goldens.npz", allow_pickle=False) as expected,
              np.load(cli_out / "consumer/host/outputs.npz", allow_pickle=False) as actual):
            assert actual.files == [f"out{i}" for i in range(7)]
            for name in actual.files:
                assert actual[name].shape == expected[name].shape and actual[name].dtype == expected[name].dtype
                assert actual[name].tobytes() == expected[name].tobytes()
