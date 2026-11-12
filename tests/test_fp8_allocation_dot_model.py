"""Composite precision and source cast contracts cross complete model bindings."""

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_fp8_allocation_dot import REQUIRED, convert

pytestmark = pytest.mark.skipif(
    not (REQUIRED and os.environ.get("T2M_MODEL2MLIR_PYTHON")),
    reason="existing model2MLIR/native/Merlin environments required",
)


@pytest.mark.parametrize("descriptors,precision", [(True, "bf16x3"), (False, "tf32x3")])
def test_complete_tiled_models_keep_original_C_precision_and_every_result(
    tmp_path, descriptors, precision
):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding, bind_model
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY

    capture = tmp_path / "capture"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/capture_fp8_allocation_dot_model.py"
    )
    process = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--out",
            str(capture),
            "--precision",
            precision,
            *(["--descriptors"] if descriptors else []),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        env=dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            TMPDIR=str(tmp_path),
        ),
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = convert(tmp_path / "source", descriptors=descriptors, precision=precision)
    bindings = []
    # One specialization serves both dependent full-K calls.
    for k in (37, 0):
        m, n = 17, 5
        values = (
            m,
            n,
            k,
            40 if k else 0,
            1,
            8,
            1,
            0,
            0,
            m * 40 if k else 0,
            40 * 8 if k else 0,
            n,
            1,
            0,
            m * n,
            0,
        )
        scalars = dict(
            zip((s.name for s in result.spec.runtime_scalars), values, strict=True)
        )
        scalars.update(
            precision_code={"bf16x3": 3, "tf32x3": 1}[precision],
            descriptors=descriptors,
        )
        bindings.append(
            KernelBinding(
                "t2m_demo::fp8_allocation_dot",
                result,
                tensor_views=POLICY,
                operands={"x": "x", "y": "weight", "state": "state"},
                scalar_arguments=scalars,
                runtime_arguments={s.name: s.name for s in result.spec.runtime_scalars},
            )
        )
    out = tmp_path / "bound"
    for name, value in (("precision_code", 4), ("descriptors", not descriptors)):
        # Bindings are matching candidates, rather than a positional call list.
        # Both full-K calls otherwise share the same scalar contract.
        wrong = [
            replace(binding, scalar_arguments={**binding.scalar_arguments, name: value})
            for binding in bindings
        ]
        with pytest.raises(ValueError, match="matching typed/scalar"):
            bind_model((capture / "model.mlir").read_text(), wrong)
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
    assert (out / "model.mlir").read_text().count('"linalg.matmul"') == 9
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
                    assert actual[name].tobytes() == expected[name].tobytes()
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_model as target

        options = {
            "merlin_root": os.environ["T2M_MERLIN_ROOT"],
            "python": os.environ["T2M_MERLIN_PYTHON"],
        }
        receipt = target.qualify(
            out, tmp_path / "target", comparison="words", **options
        )
        assert target.verify(tmp_path / "target", **options) == receipt
        assert receipt["evaluation"]["result_count"] == 4
