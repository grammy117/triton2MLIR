"""Model2MLIR retains aliased typed-source state and every public result."""

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


@pytest.mark.parametrize("typed_root", (False, True))
def test_real_word_model_preserves_weights_alias_recurrence_and_all_results(tmp_path, typed_root):
    from t2m.allocation_word_demo import convert
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY

    capture = tmp_path / "capture"
    script = (
        Path(__file__).resolve().parents[1] / "scripts/capture_allocation_word_model.py"
    )
    process = subprocess.run(
        [os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(capture), *(["--typed-root"] if typed_root else [])],
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
    result = convert(tmp_path / "source", root_dtype="i64", byte_order="little") if typed_root else convert(tmp_path / "source")
    names = [scalar.name for scalar in result.spec.runtime_scalars]
    bindings = [
        KernelBinding(
            "t2m_demo::allocation_words",
            result,
            tensor_views=POLICY,
            scalar_arguments=dict(zip(names, values, strict=True)),
            runtime_arguments={name: name for name in names},
        )
        for values in ((7, 2, 96, 8, 10), (31, 3, 96, 8, 8), (0, 2, 0, 511, -512))
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
                assert actual.files == ["out0", "out1", "out2"]
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
    if typed_root:
        from test_typed_parameter_cli import qualify_model_cli
        qualify_model_cli(tmp_path, capture, bindings, result)
