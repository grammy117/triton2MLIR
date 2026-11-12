"""Complete half-atomic models preserve state, source words and six results."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_half_atomic_add import MERLIN, NATIVE, OPT, capture

pytestmark = pytest.mark.skipif(
    not (MERLIN and NATIVE and OPT and os.environ.get("T2M_MODEL2MLIR_PYTHON")),
    reason="existing native/model2MLIR/Merlin tools required",
)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_complete_half_model_keeps_every_result_and_live_original_state(
    tmp_path, dtype
):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_words import TensorWords

    parent = tmp_path / "parent"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/capture_half_atomic_model.py"
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
        text=True,
        timeout=60,
        env=dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            TMPDIR=str(tmp_path),
        ),
    )
    assert process.returncode == 0, process.stdout + process.stderr
    source = tmp_path / "source"
    source.mkdir()
    result = capture(source, dtype)
    bindings = [
        KernelBinding(
            "t2m_demo::half_atomic_state",
            result,
            scalar_arguments={"n": n, "m": 37},
            runtime_arguments={"n": "n", "m": "m"},
        )
        for n in (31, 0)
    ]
    out = tmp_path / "bound"
    write_model_bundle(
        parent,
        bindings,
        out,
        work=tmp_path / "qualification",
        python=NATIVE,
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
        np.load(parent / "goldens.npz", allow_pickle=False) as golden,
    ):
        expected = [
            words.get(
                "goldens.npz",
                f"out{i}",
                golden[f"out{i}"],
                dtype if i in (0, 1, 5) else "i16",
                golden[f"out{i}"].shape,
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
                    actual[f"out{i}"].dtype == value.dtype
                    and actual[f"out{i}"].shape == value.shape
                )
                assert actual[f"out{i}"].tobytes() == value.tobytes()
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_model as target

        options = dict(
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        report = target.qualify(out, tmp_path / "target", comparison="words", **options)
        assert target.verify(tmp_path / "target", **options) == report
        assert report["evaluation"]["result_count"] == 6
