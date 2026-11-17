"""Actual model controls and shape SSA stay live through reusable Merlin code."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_pointer_addresses import REQUIRED

from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not REQUIRED or not os.environ.get("T2M_MODEL2MLIR_PYTHON"),
                                reason="existing model2MLIR/native/Merlin tools required")


def implementation(root, dtype="i64", *, extent=False, profile="guarded-buffer"):
    from t2m.examples.live_scalar_kernels import live_extent_copy, live_scalar_values

    kernel = live_extent_copy if extent else live_scalar_values
    word = "i32" if dtype == "i1" else dtype
    source_type = {"f32": "fp32", "f64": "fp64"}.get(word, word)
    control_type = {"f32": "fp32", "f64": "fp64"}.get(dtype, dtype)
    constants = {"B": 8} if extent else {"B": 8, "N": 13,
        "KIND": "bool" if dtype == "i1" else "float" if dtype.startswith("f") else "integer"}
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, {"x": "*" + source_type, "out": "*" + source_type,
        "n" if extent else "control": "i32" if extent else control_type,
        **{name: "constexpr" for name in constants}}, constants))
    shape = ("n",) if extent else (13,)
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", shape, dtype=word),
        TensorBuffer("out", shape, "write", word)), grid=(GridExtent("n", 8),) if extent else (2,),
        runtime_scalars=(RuntimeScalar("n", "i32", 1, bounds=(0, 31)) if extent else RuntimeScalar("control", dtype, 1),))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
                          memory_profile=profile, address_policy="source-words" if profile == "guarded-buffer" else "structured")


def capture_and_bind(root, dtype, *, extent=False, profile="guarded-buffer"):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle

    parent = root / "parent"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_live_scalar_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(parent),
        "--dtype", dtype, *(["--extent"] if extent else [])], capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert process.returncode == 0, process.stdout + process.stderr
    result = implementation(root, dtype, extent=extent, profile=profile)
    name = "t2m_demo::live_extent" if extent else "t2m_demo::live_scalar_" + dtype
    bindings = [KernelBinding(name, result, runtime_operands={"n" if extent else "control": "n" if extent else "control"})]
    if extent:
        bindings.append(KernelBinding(name, result, scalar_arguments={"n": 0}, runtime_arguments={"n": "n"}))
    out = root / "bound"
    write_model_bundle(parent, bindings, out, work=root / "qualification", python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"], merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_host=True, merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(out)
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert trace["argument_binding_profile"] == "t2m.live-runtime-arguments.v1" and trace["model_abi"] == "unchanged"
    assert len(trace["calls"]) == (3 if extent else 2) and len(trace["kernels"]) == 1
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (parent / name).read_bytes()
    with np.load(parent / "goldens.npz", allow_pickle=False) as expected:
        assert expected.files == [f"out{i}" for i in range(5)]
        for path in (root / "qualification/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == expected.files
                for name in expected.files:
                    assert (actual[name].shape, actual[name].dtype, actual[name].tobytes()) == (
                        expected[name].shape, expected[name].dtype, expected[name].tobytes())
    return out


@pytest.mark.parametrize("dtype", ["i1", "i8", "i16", "i32", "i64", "f32", "f64"])
def test_complete_model_controls_remain_runtime_values_in_unchanged_merlin_library(tmp_path, dtype):
    from scripts import qualify_spike_model as target
    from t2m.bundle import _record
    from t2m.consumer import invoke_merlin_host

    out = capture_and_bind(tmp_path, dtype)
    execution = json.loads((out / "execution.json").read_text())
    sources = [item["storage"] for item in execution["materialized_arguments"]]
    with np.load(tmp_path / "qualification/arguments/arguments.npz", allow_pickle=False) as stored:
        arrays = [stored[f"in{i}"].copy() for i in range(len(stored.files))]
    x_index, control_index, weight_index = (sources.index(name) for name in ("inputs:in0", "inputs:in1", "weights:weight"))
    original = arrays[x_index].copy()
    controls = ([False, True, False, True] if dtype == "i1" else [0.0, -.25, 1.5, -0.0] if dtype.startswith("f") else
                [0, -1, -(1 << (int(dtype[1:]) - 1)), (1 << (int(dtype[1:]) - 1)) - 1])
    library = _record(out / "consumer/host/model.so")

    def apply(value, control):
        if dtype == "i1":
            return value.copy() if bool(control) else -value
        if dtype.startswith("f"):
            return np.add(value, control, dtype=value.dtype)
        return np.bitwise_xor(value, control)

    for index, control in enumerate(controls):
        arrays[x_index] = original.copy() if index % 2 == 0 else original[::-1].copy()
        arrays[control_index] = np.full((1, 1), control, arrays[control_index].dtype)
        first = apply(arrays[x_index], arrays[weight_index].reshape(-1)[0])
        second = apply(first, arrays[control_index].reshape(-1)[0])
        expected = [second, first, arrays[x_index], arrays[control_index], arrays[weight_index]]
        caller = tmp_path / f"caller-{index}.npz"
        np.savez(caller, **{f"in{i}": value for i, value in enumerate(arrays)})
        actual, report = invoke_merlin_host(out / "consumer", caller, tmp_path / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for value, reference in zip(actual, expected, strict=True):
            assert (value.shape, value.dtype, value.tobytes()) == (reference.shape, reference.dtype, reference.tobytes())
        assert _record(out / "consumer/host/model.so") == library
    if dtype == "i64":
        from t2m.bundle import verify_bundle
        from t2m.cli import main
        spec = json.loads((out / "kernels/0/spec.json").read_text())
        configuration = {"schema": "t2m.kernel-bindings.v1", "kernels": [{"kernel": "t2m_demo::live_scalar_i64",
            "ttir": "source.ttir", "abi": spec, "runtime_operands": {"control": "control"},
            "memory_profile": "guarded-buffer", "address_policy": "source-words"}]}
        (tmp_path / "bindings.json").write_text(json.dumps(configuration))
        cli = tmp_path / "cli-bound"
        assert main(["bind-model", str(tmp_path / "parent"), "--bindings", str(tmp_path / "bindings.json"),
            "--out", str(cli), "--optimizer", os.environ["TRITON_SHARED_OPT"], "--python", os.environ["T2M_NATIVE_PYTHON"],
            "--merlin-root", os.environ["T2M_MERLIN_ROOT"], "--merlin-python", os.environ["T2M_MERLIN_PYTHON"],
            "--merlin-clang", os.environ["T2M_MERLIN_CLANG"], "--merlin-host"]) == 0
        verify_bundle(cli)
        assert json.loads((cli / "kernel-bindings.json").read_text())["argument_binding_profile"] == "t2m.live-runtime-arguments.v1"
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    report = target.qualify(out, tmp_path / "target", comparison="words", **options)
    assert target.verify(tmp_path / "target", **options) == report and report["evaluation"]["result_count"] == 5


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_actual_exported_shape_operands_and_captured_empty_call_share_one_body(tmp_path, profile):
    from scripts import qualify_spike_model as target

    out = capture_and_bind(tmp_path, "i32", extent=True, profile=profile)
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert [bool(call.get("runtime_operand_bindings")) for call in trace["calls"]] == [True, True, False]
    assert all(call["runtime_operand_bindings"]["n"]["constant_integer"] == 13 for call in trace["calls"][:2])
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    report = target.qualify(out, tmp_path / "target", comparison="words", **options)
    assert target.verify(tmp_path / "target", **options) == report and report["evaluation"]["result_count"] == 5
