"""Whole models keep coupled result tuples and live original state through Merlin."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_reduction_kernels import REQUIRED, capture

pytestmark = pytest.mark.skipif(
    not (REQUIRED and os.environ.get("T2M_MODEL2MLIR_PYTHON")),
    reason="existing model2MLIR/native/Merlin environments required",
)


@pytest.mark.parametrize("kind", ["affine", "argmin"])
@pytest.mark.parametrize("axis", [0, 1])
def test_complete_reduction_model_retains_every_original_result(tmp_path, kind, axis):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle

    parent = tmp_path / "parent"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_reduction_model.py"
    process = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--out",
            str(parent),
            "--kind",
            kind,
            "--axis",
            str(axis),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=dict(
            os.environ,
            TMPDIR=str(tmp_path),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        ),
    )
    assert process.returncode == 0, process.stdout + process.stderr
    source = tmp_path / "source"
    source.mkdir()
    result = capture(source, kind, axis)
    bindings = [
        KernelBinding(
            "t2m_demo::reduction_state",
            result,
            scalar_arguments={"n": n, "steps": steps, "kind": kind, "axis": axis},
            runtime_arguments={"n": "n", "steps": "steps"},
        )
        for n, steps in [(19, 1), (19, 2), (0, 3)]
    ]
    out = tmp_path / "bound"
    write_model_bundle(
        parent,
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
    for name in (
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "golden.npy",
        "goldens.npz",
        "extra.npz",
    ):
        assert (parent / name).read_bytes() == (out / name).read_bytes()
    with np.load(parent / "goldens.npz", allow_pickle=False) as expected:
        assert expected.files == [f"out{i}" for i in range(8)]
        for path in (
            tmp_path / "qualification/native/outputs.npz",
            out / "consumer/host/outputs.npz",
        ):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == expected.files
                for name in actual.files:
                    assert (
                        actual[name].shape,
                        actual[name].dtype,
                        actual[name].tobytes(),
                    ) == (
                        expected[name].shape,
                        expected[name].dtype,
                        expected[name].tobytes(),
                    )
    from scripts import qualify_spike_model as target

    options = {
        "merlin_root": os.environ["T2M_MERLIN_ROOT"],
        "python": os.environ["T2M_MERLIN_PYTHON"],
    }
    report = target.qualify(out, tmp_path / "target", comparison="words", **options)
    assert target.verify(tmp_path / "target", **options) == report
    assert report["evaluation"]["result_count"] == 8
