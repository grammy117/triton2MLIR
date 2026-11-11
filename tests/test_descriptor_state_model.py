"""Original descriptor intent survives complete model2MLIR/Merlin composition."""

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


def test_descriptor_views_bind_real_model_values_weights_and_complete_results(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.descriptor_state_demo import convert
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY
    from t2m.source_capsule import members

    capture = tmp_path / "capture"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/capture_descriptor_state_model.py"
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
    result = convert(tmp_path / "source", dtype="i64")
    scalars = [scalar.name for scalar in result.spec.runtime_scalars]
    bindings = [
        KernelBinding(
            "t2m_demo::descriptor_state",
            result,
            tensor_views=POLICY,
            scalar_arguments=dict(zip(scalars, values, strict=True)),
            runtime_arguments={name: name for name in scalars},
        )
        for values in (
            (2, 13, 2, 256, 8, 16, 24, 1, 24, 1),
            (5, 19, 3, 256, 8, 8, 24, 1, 24, 1),
            (0, 13, 2, 0, 8191, 8191, 24, 1, 24, 1),
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
        trace["model_abi"] == "unchanged"
        and len(trace["calls"]) == 3
        and len(trace["kernels"]) == 1
    )
    for name in (
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "golden.npy",
        "goldens.npz",
        "extra.npz",
    ):
        assert (capture / name).read_bytes() == (out / name).read_bytes()
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected:
        for path in (
            tmp_path / "qualification/native/outputs.npz",
            out / "consumer/host/outputs.npz",
        ):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == ["out0", "out1", "out2"]
                for name in actual.files:
                    assert (
                        actual[name].dtype == expected[name].dtype
                        and actual[name].shape == expected[name].shape
                    )
                    assert actual[name].tobytes() == expected[name].tobytes()
    for name in members(result.trace):
        artifacts = list((out / "kernels").glob("*/" + name))
        staged = list((out / "consumer/compiler/source/kernels").glob("*/" + name))
        assert len(artifacts) == len(staged) == 1
        assert artifacts[0].read_bytes() == staged[0].read_bytes()
    host = json.loads((out / "consumer/host/execution.json").read_text())
    assert (
        host["whole_body_lowering"]
        == host["compiler_input_staging"]
        == host["reference_comparison"]
        == "passed"
    )
    from t2m.package_coverage import package_coverage

    report = package_coverage([out])
    assert report["ttir"]["tt.descriptor_load"]["merlin_cases"]
    assert (
        report["kernel_cases"][0]["source_graph"]
        == "pre-wrapper-original-source-operations"
    )
