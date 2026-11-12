"""Every original model result retains source literal words and snapshots."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_literal_state import REQUIRED, capture

pytestmark = pytest.mark.skipif(
    not (REQUIRED and os.environ.get("T2M_MODEL2MLIR_PYTHON")),
    reason="existing model2MLIR/native/Merlin tools required",
)


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
def test_complete_literal_model_keeps_all_six_results_and_original_state(
    tmp_path, dtype
):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_words import TensorWords

    parent = tmp_path / "parent"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/capture_literal_state_model.py"
    )
    process = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--out",
            str(parent),
            "--dtype",
            dtype,
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
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
    result = capture(source, dtype, allocated=True)
    bindings = [
        KernelBinding(
            "t2m_demo::literal_state",
            result,
            scalar_arguments={"n": n, "steps": steps},
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
        *(("tensor_words.npz",) if dtype == "bf16" else ()),
    ):
        assert (parent / name).read_bytes() == (out / name).read_bytes()
    with (
        TensorWords(parent) as words,
        np.load(parent / "goldens.npz", allow_pickle=False) as goldens,
    ):
        expected = [
            words.get(
                "goldens.npz",
                f"out{i}",
                goldens[f"out{i}"],
                dtype,
                goldens[f"out{i}"].shape,
            )
            for i in range(6)
        ]
    for path in (
        tmp_path / "qualification/native/outputs.npz",
        out / "consumer/host/outputs.npz",
    ):
        with np.load(path, allow_pickle=False) as actual:
            assert actual.files == [f"out{i}" for i in range(6)]
            for i, value in enumerate(expected):
                assert (
                    actual[f"out{i}"].shape,
                    actual[f"out{i}"].dtype,
                    actual[f"out{i}"].tobytes(),
                ) == (value.shape, value.dtype, value.tobytes())
    from scripts import qualify_spike_model as target

    options = {
        "merlin_root": os.environ["T2M_MERLIN_ROOT"],
        "python": os.environ["T2M_MERLIN_PYTHON"],
    }
    report = target.qualify(out, tmp_path / "target", comparison="words", **options)
    assert target.verify(tmp_path / "target", **options) == report
    assert report["evaluation"]["result_count"] == 6
