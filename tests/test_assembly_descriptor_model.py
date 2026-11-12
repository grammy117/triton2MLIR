"""Complete learned word models retain original assembly and descriptor state."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_assembly_descriptor_intake import PROVIDER, REQUIRED, convert

pytestmark = pytest.mark.skipif(
    not (REQUIRED and PROVIDER and os.environ.get("T2M_MODEL2MLIR_PYTHON")),
    reason="existing model/native/Merlin and source provider required",
)


@pytest.mark.parametrize("dtype", ["i32", "i64"])
def test_decoded_tuple_arithmetic_and_descriptor_state_keep_every_model_result(
    tmp_path, dtype
):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY
    from t2m.source_capsule import members

    capture = tmp_path / "capture"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/capture_assembly_descriptor_model.py"
    )
    process = subprocess.run(
        [
            os.environ["T2M_MODEL2MLIR_PYTHON"],
            str(script),
            "--out",
            str(capture),
            "--dtype",
            dtype,
        ],
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
    result = convert(tmp_path / "source", dtype=dtype, words=True)
    names = [s.name for s in result.spec.runtime_scalars]
    bindings = [
        KernelBinding(
            "t2m_demo::assembly_descriptor_state",
            result,
            tensor_views=POLICY,
            scalar_arguments=dict(zip(names, values, strict=True)),
            runtime_arguments={name: name for name in names},
        )
        for values in (
            (2, 13, 2, 256, 8, 16, 24, 1, 24, 1, 3),
            (5, 19, 3, 256, 8, 8, 24, 1, 24, 1, 33),
            (0, 13, 2, 0, 8191, 8191, 24, 1, 24, 1, -1),
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
    for name in members(result.trace):
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
