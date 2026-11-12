"""Complete captured model state, source contracts and all results reach Merlin."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_fp8_allocation_intake import REQUIRED, convert

pytestmark = pytest.mark.skipif(
    not (REQUIRED and os.environ.get("T2M_MODEL2MLIR_PYTHON")),
    reason="existing model2MLIR/native/Merlin environments required",
)


def test_protected_descriptor_recurrence_retains_model_weights_snapshots_and_all_results(
    tmp_path,
):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY
    from t2m.source_capsule import members

    capture = tmp_path / "capture"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/capture_fp8_allocation_model.py"
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
    result = convert(tmp_path / "source")
    scalars = [scalar.name for scalar in result.spec.runtime_scalars]
    bindings = [
        KernelBinding(
            "t2m_demo::fp8_allocation",
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
    binding = json.loads((out / "kernel-bindings.json").read_text())
    assert (
        binding["model_abi"] == "unchanged"
        and len(binding["calls"]) == 3
        and len(binding["kernels"]) == 1
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
        assert expected.files == ["out0", "out1", "out2", "out3"]
        for path in (
            tmp_path / "qualification/native/outputs.npz",
            out / "consumer/host/outputs.npz",
        ):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == expected.files
                for name in actual.files:
                    assert (
                        actual[name].dtype == expected[name].dtype
                        and actual[name].shape == expected[name].shape
                    )
                    np.testing.assert_array_equal(actual[name], expected[name])
                assert actual["out3"].tobytes() == expected["out3"].tobytes()
    for name in members(result.trace):
        (source,) = (out / "kernels").glob("*/" + name)
        (staged,) = (out / "consumer/compiler/source/kernels").glob("*/" + name)
        assert source.read_bytes() == staged.read_bytes()
    from t2m.package_coverage import package_coverage

    report = package_coverage([out])
    (case,) = report["kernel_cases"]
    assert case["source_graph"] == "pre-wrapper-original-source-operations"
    assert (
        case["allocation_bound_operation_counts"]["tt.call"] == 2
    )  # Wrapper and private FP8 helper.
    assert (
        sum(
            len(form["observations"]) for form in report["ttir"]["tt.fp_to_fp"]["forms"]
        )
        == 6
    )
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_model as target

        options = {
            "merlin_root": os.environ["T2M_MERLIN_ROOT"],
            "python": os.environ["T2M_MERLIN_PYTHON"],
        }
        receipt = target.qualify(out, tmp_path / "target", word_results=(3,), **options)
        assert target.verify(tmp_path / "target", **options) == receipt
        assert receipt["evaluation"]["result_count"] == 4
