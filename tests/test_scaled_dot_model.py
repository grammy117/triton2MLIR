"""Actual model2MLIR capture retains scaled contractions and complete tuples."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from t2m.bundle import verify_bundle
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE, M2M = (
    os.environ.get(key)
    for key in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON")
)
MERLIN, CLANG = os.environ.get("T2M_MERLIN_ROOT"), os.environ.get("T2M_MERLIN_CLANG")
pytestmark = pytest.mark.skipif(
    not (OPT and NATIVE and M2M),
    reason="existing shared/native/model2MLIR environments required",
)


@pytest.fixture(scope="module")
def model_capture(tmp_path_factory):
    root = tmp_path_factory.mktemp("scaled-model")
    script = Path(__file__).resolve().parents[1] / "scripts/capture_scaled_dot_model.py"
    process = subprocess.run(
        [M2M, str(script), "--out", str(root / "capture")],
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(
            os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)
        ),
    )
    assert process.returncode == 0, process.stderr
    return root / "capture"


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_scaled_kernel_binds_actual_model_weights_and_complete_outputs(
    tmp_path, model_capture, profile
):
    from t2m.examples.scaled_dot_kernels import scaled_dot_tile

    constants = dict(
        M=16,
        N=16,
        K=64,
        A_FORMAT="e2m1",
        B_FORMAT="e2m1",
        LHS_K_PACK=False,
        RHS_K_PACK=False,
        HAS_A_SCALE=True,
        HAS_B_SCALE=True,
        GROUP=32,
    )
    signature = {
        name: "*u8" if i < 4 else "*fp32" if i < 6 else "constexpr"
        for i, name in enumerate(scaled_dot_tile.arg_names)
    }
    source = tmp_path / "source.ttir"
    source.write_text(
        capture_ttir(
            scaled_dot_tile, signature, constants, source_target=("hip", "gfx942", 64)
        )
    )
    abi = TensorABI(
        "scaled_dot_tile",
        (
            TensorBuffer("a", (8, 64), dtype="i8"),
            TensorBuffer("b", (64, 8), dtype="i8"),
            TensorBuffer("sa", (16, 2), dtype="i8"),
            TensorBuffer("sb", (16, 2), dtype="i8"),
            TensorBuffer("initial", (16, 16)),
            TensorBuffer("out", (16, 16), "write"),
        ),
    )
    result = convert_shared(
        source,
        abi,
        tmp_path / "convert",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    binding = KernelBinding(
        "t2m_demo::scaled_dot",
        result,
        scalar_arguments={"lhs_k_pack": False, "rhs_k_pack": False},
    )
    qualify_model(tmp_path, model_capture, binding)


def qualify_model(tmp_path, model_capture, binding):
    out = tmp_path / "bound"
    write_model_bundle(
        model_capture,
        [binding],
        out,
        work=tmp_path / "work",
        python=NATIVE,
        merlin_root=MERLIN,
        merlin_python=os.environ.get("T2M_MERLIN_PYTHON"),
        merlin_host=bool(MERLIN and CLANG),
        merlin_clang=CLANG,
        rtol=0,
        atol=0,
    )
    receipt = verify_bundle(out)
    for name in (
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "input_order.json",
        "golden.npy",
        "goldens.npz",
        "extra.npz",
    ):
        assert (out / name).read_bytes() == (model_capture / name).read_bytes()
    model = (out / "model.mlir").read_text()
    assert "prov.scaled_dot_contracts" in model
    assert '"func.call"' not in model
    execution = json.loads((out / "execution.json").read_text())
    assert execution["result_count"] == 2
    assert set(execution["result_reference_members"]) == {"golden.npy", "goldens.npz"}
    assert any(
        "logical-mx-f64-v1" in item
        for item in json.loads((out / "compatibility.json").read_text())[
            "kernel_profiles"
        ]["kernels/0"]["differences"]
    )
    assert "parent/model.mlir" in receipt["artifacts"]
    if binding.runtime_arguments:
        calls = json.loads((out / "kernel-bindings.json").read_text())["calls"]
        assert set(calls[0]["runtime_scalar_bindings"]) == set(
            binding.runtime_arguments
        )
        assert any(
            item["kind"] == "result-shape-cast"
            for item in calls[0]["argument_adaptations"]
        )
    if MERLIN and CLANG:
        host = json.loads((out / "consumer/host/execution.json").read_text())
        assert host["host_execution"] == host["whole_body_lowering"] == "passed"
        assert host["evaluation"]["functional_inputs_preserved"]
        assert host["result_count"] == 2
        with (
            np.load(model_capture / "goldens.npz", allow_pickle=False) as reference,
            np.load(out / "consumer/host/outputs.npz", allow_pickle=False) as actual,
        ):
            for index in range(2):
                np.testing.assert_array_equal(
                    actual[f"out{index}"].view(np.uint32),
                    reference[f"out{index}"].view(np.uint32),
                )


@pytest.fixture(scope="module")
def dynamic_capture(tmp_path_factory):
    root = tmp_path_factory.mktemp("scaled-model-dynamic")
    script = Path(__file__).resolve().parents[1] / "scripts/capture_scaled_dot_model.py"
    process = subprocess.run(
        [M2M, str(script), "--dynamic", "--out", str(root / "capture")],
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(
            os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)
        ),
    )
    assert process.returncode == 0, process.stderr
    return root / "capture"


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_ragged_runtime_kernel_binds_captured_scalars_and_preserves_model_abi(
    tmp_path, dynamic_capture, profile
):
    from test_scaled_dot_dynamic import convert

    result = convert(tmp_path, profile)
    arguments = {"m": 17, "n": 19, "k": 37, "packed_k": 19, "groups": 2}
    binding = KernelBinding(
        "t2m_demo::scaled_dot_dynamic",
        result,
        scalar_arguments=arguments,
        runtime_arguments={name: name for name in arguments},
    )
    qualify_model(tmp_path, dynamic_capture, binding)


@pytest.mark.parametrize("dynamic", [False, True])
def test_model_signed_word_carriers_capture_native_verified_packed_ttir(dynamic):
    from t2m.examples.scaled_dot_kernels import scaled_dot_dynamic, scaled_dot_tile

    kernel = scaled_dot_dynamic if dynamic else scaled_dot_tile
    constants = (
        {"BM": 16, "BN": 16}
        if dynamic
        else dict(
            M=16,
            N=16,
            K=64,
            A_FORMAT="e2m1",
            B_FORMAT="e2m1",
            LHS_K_PACK=False,
            RHS_K_PACK=False,
            HAS_A_SCALE=True,
            HAS_B_SCALE=True,
            GROUP=32,
        )
    )
    signature = {
        name: "*i8"
        if i < 4
        else "*fp32"
        if i < 6
        else "constexpr"
        if name in constants
        else "i32"
        for i, name in enumerate(kernel.arg_names)
    }
    source = capture_ttir(
        kernel, signature, constants, source_target=("hip", "gfx942", 64)
    )
    assert "tt.dot_scaled" in source and "!tt.ptr<i8>" in source
