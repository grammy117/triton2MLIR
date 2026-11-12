"""Whole models keep learned state, dependent structured scans and every result."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_scan_control import MERLIN, NATIVE, OPT, capture, loop_capture

pytestmark = pytest.mark.skipif(
    not (MERLIN and NATIVE and OPT and os.environ.get("T2M_MODEL2MLIR_PYTHON")),
    reason="existing native/model2MLIR/Merlin tools required",
)


@pytest.mark.parametrize("gcd", [False, True])
def test_complete_model_keeps_branch_and_loop_scans_and_original_seven_results(
    tmp_path, gcd
):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle

    parent = tmp_path / "parent"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/capture_scan_control_model.py"
    )
    process = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--out",
            str(parent),
            *(["--gcd"] if gcd else []),
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
    bindings = []
    for reverse in (False, True):
        root = tmp_path / f"segments-{reverse}"
        root.mkdir()
        result = capture(root, "i64", reverse)
        bindings.append(
            KernelBinding(
                "t2m_demo::segmented_scan",
                result,
                scalar_arguments={"n": 19, "reverse": reverse},
                runtime_arguments={"n": "n"},
            )
        )
    root = tmp_path / "loops"
    root.mkdir()
    result = loop_capture(root, 64, False, gcd)
    bindings.extend(
        KernelBinding(
            "t2m_demo::loop_scan",
            result,
            scalar_arguments={"n": n, "gcd": gcd},
            runtime_arguments={"n": "n"},
        )
        for n in (19, 0)
    )
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
        assert (out / name).read_bytes() == (parent / name).read_bytes()
    with np.load(parent / "goldens.npz", allow_pickle=False) as expected:
        for path in (
            tmp_path / "qualification/native/outputs.npz",
            out / "consumer/host/outputs.npz",
        ):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == expected.files == [f"out{i}" for i in range(7)]
                for name in expected.files:
                    assert (
                        actual[name].shape == expected[name].shape
                        and actual[name].dtype == expected[name].dtype
                    )
                    assert actual[name].tobytes() == expected[name].tobytes()
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_model as target

        options = dict(
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        report = target.qualify(out, tmp_path / "target", comparison="words", **options)
        assert target.verify(tmp_path / "target", **options) == report
        assert report["evaluation"]["result_count"] == 7
