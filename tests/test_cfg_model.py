"""Whole-model CFG intake preserves learned parameters and every result."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_cfg_intake import MERLIN, NATIVE, OPT, capture

pytestmark = pytest.mark.skipif(
    not (MERLIN and NATIVE and OPT and os.environ.get("T2M_MODEL2MLIR_PYTHON")),
    reason="existing model/native/Merlin tools required",
)


def test_complete_learned_model_retains_original_cfg_and_all_four_results(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.cfg_intake import FILES
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle

    parent = tmp_path / "parent"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_cfg_model.py"
    process = subprocess.run(
        [os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(parent)],
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
    source = tmp_path / "source"
    source.mkdir()
    result = capture(source)
    names = [value.name for value in result.spec.runtime_scalars]
    bindings = [
        KernelBinding(
            "t2m_demo::cfg_state",
            result,
            scalar_arguments=dict(zip(names, values, strict=True)),
            runtime_arguments={name: name for name in names},
        )
        for values in ((19, 3, 0), (19, 2, 1), (0, 1, 1))
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
    ):
        assert (parent / name).read_bytes() == (out / name).read_bytes()
    with np.load(parent / "goldens.npz", allow_pickle=False) as expected:
        for path in (
            tmp_path / "qualification/native/outputs.npz",
            out / "consumer/host/outputs.npz",
        ):
            with np.load(path, allow_pickle=False) as actual:
                assert (
                    actual.files == expected.files == ["out0", "out1", "out2", "out3"]
                )
                for name in actual.files:
                    assert (
                        actual[name].dtype == expected[name].dtype
                        and actual[name].shape == expected[name].shape
                    )
                    assert actual[name].tobytes() == expected[name].tobytes()
    for name in FILES:
        (original,) = (out / "kernels").glob("*/" + name)
        (staged,) = (out / "consumer/compiler/source/kernels").glob("*/" + name)
        assert original.read_bytes() == staged.read_bytes()
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
