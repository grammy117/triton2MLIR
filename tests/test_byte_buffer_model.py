"""Byte-backed typed events compose through actual model2MLIR/Merlin captures."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_descriptor_state import REQUIRED

pytestmark = pytest.mark.skipif(
    not (REQUIRED and os.environ.get("T2M_MODEL2MLIR_PYTHON")),
    reason="existing model2MLIR/native/Merlin environments required",
)


def test_real_byte_cache_model_keeps_five_results_weights_and_state(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.byte_buffer_demo import convert
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY

    capture = tmp_path / "capture"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/capture_byte_buffer_model.py"
    )
    process = subprocess.run(
        [os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            TMPDIR=str(tmp_path),
        ),
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = convert(tmp_path / "source", dtype="f32")
    names = [scalar.name for scalar in result.spec.runtime_scalars]
    bindings = [
        KernelBinding(
            "t2m_demo::byte_cache",
            result,
            tensor_views=POLICY,
            operands={"raw": "state"},
            scalar_arguments=dict(zip(names, values, strict=True)),
            runtime_arguments={name: name for name in names},
        )
        for values in (
            (7, 2, 256, 9, 24, 1),
            (13, 1, 256, 7, 16, 0),
            (0, 3, 0, 8191, -128, -64),
        )
    ]
    out = tmp_path / "bound"
    write_model_bundle(
        capture,
        bindings,
        out,
        work=tmp_path / "qualification",
        python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_host=True,
        merlin_clang=os.environ["T2M_MERLIN_CLANG"],
    )
    verify_bundle(out)
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert (
        len(trace["calls"]) == 3
        and len(trace["kernels"]) == 1
        and trace["model_abi"] == "unchanged"
    )
    for name in (
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "golden.npy",
        "goldens.npz",
        "extra.npz",
    ):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected:
        for path in (
            tmp_path / "qualification/native/outputs.npz",
            out / "consumer/host/outputs.npz",
        ):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == ["out0", "out1", "out2", "out3", "out4"]
                for name in actual.files:
                    assert (
                        actual[name].dtype == expected[name].dtype
                        and actual[name].shape == expected[name].shape
                    )
                    assert actual[name].tobytes() == expected[name].tobytes()
    host = json.loads((out / "consumer/host/execution.json").read_text())
    assert (
        host["whole_body_lowering"]
        == host["compiler_input_staging"]
        == host["reference_comparison"]
        == "passed"
    )
    from t2m.cli import main

    configuration = {
        "schema": "t2m.kernel-bindings.v1",
        "kernels": [
            {
                "kernel": binding.qualname,
                "ttir": "source/source.ttir",
                "abi": result.spec.as_dict(),
                "operands": binding.operands,
                "tensor_views": POLICY,
                "scalar_arguments": binding.scalar_arguments,
                "runtime_arguments": binding.runtime_arguments,
                "memory_profile": "guarded-buffer",
                "address_policy": "source-words",
                "byte_order": "little",
            }
            for binding in bindings
        ],
    }
    (tmp_path / "bindings.json").write_text(json.dumps(configuration))
    cli_out = tmp_path / "cli-bound"
    assert (
        main(
            [
                "bind-model",
                str(capture),
                "--bindings",
                str(tmp_path / "bindings.json"),
                "--out",
                str(cli_out),
                "--optimizer",
                os.environ["TRITON_SHARED_OPT"],
                "--python",
                os.environ["T2M_NATIVE_PYTHON"],
                "--merlin-root",
                os.environ["T2M_MERLIN_ROOT"],
                "--merlin-python",
                os.environ["T2M_MERLIN_PYTHON"],
                "--merlin-clang",
                os.environ["T2M_MERLIN_CLANG"],
                "--merlin-host",
            ]
        )
        == 0
    )
    verify_bundle(cli_out)
