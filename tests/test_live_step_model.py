"""A live scalar tensor controls original address loops, effects and guards."""

import json
import os
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_integer_address_regions import capture
from test_pointer_addresses import REQUIRED

pytestmark = pytest.mark.skipif(not REQUIRED or not os.environ.get("T2M_MODEL2MLIR_PYTHON"),
                                reason="existing model2MLIR/native/Merlin tools required")


def test_live_step_tensor_retains_loop_state_guards_snapshots_and_library_reuse(tmp_path):
    from scripts import qualify_spike_model as target
    from t2m.bundle import _record, verify_bundle
    from t2m.consumer import invoke_merlin_host, observe_merlin_host
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle

    parent = tmp_path / "parent"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_live_step_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(parent)],
        capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stdout + process.stderr
    result = capture(tmp_path, "alternating", 64)
    names = ("n", "capacity", "origin", "stride", "choice")
    bindings = [KernelBinding("t2m_demo::live_step_state", result,
        operands={"x": "x", "y": "y", "state": "initial"},
        scalar_arguments=dict(zip(names, values, strict=True)), runtime_arguments={name: name for name in names},
        runtime_operands={"steps": "steps"}) for values in ((19, 128, 80, -3, 1), (19, 128, 31, -2, -1))]
    out = tmp_path / "bound"
    write_model_bundle(parent, bindings, out, work=tmp_path / "qualification", python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"], merlin_python=os.environ["T2M_MERLIN_PYTHON"], merlin_host=True,
        merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(out)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz",
                 "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (parent / name).read_bytes()
    with np.load(parent / "goldens.npz", allow_pickle=False) as expected:
        assert expected.files == [f"out{i}" for i in range(7)]
        for path in (tmp_path / "qualification/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == expected.files
                for name in expected.files:
                    assert (actual[name].shape, actual[name].dtype, actual[name].tobytes()) == (
                        expected[name].shape, expected[name].dtype, expected[name].tobytes())
    execution = json.loads((out / "execution.json").read_text())
    sources = [item["storage"] for item in execution["materialized_arguments"]]
    with np.load(tmp_path / "qualification/arguments/arguments.npz", allow_pickle=False) as stored:
        arrays = [stored[f"in{i}"].copy() for i in range(len(stored.files))]
    x_index, step_index, weight_index, initial_index = (sources.index(name) for name in
        ("inputs:in0", "inputs:in1", "weights:weight", "weights:initial"))
    original_x = arrays[x_index].copy()
    library = _record(out / "consumer/host/model.so")

    def update(x, y, initial, steps, origin, stride, choice):
        result = initial.copy()
        position, root = origin, x
        for iteration in range(steps):
            result = np.bitwise_xor(result, root[position:position + len(result)])
            position += stride
            root = x if (iteration + choice) % 2 == 0 else y
        return result, np.full(len(result), steps * stride * 8, np.int64)

    for index, steps in enumerate((0, 1, 2, 4)):
        arrays[x_index] = original_x.copy() if index % 2 == 0 else original_x[::-1].copy()
        arrays[step_index] = np.asarray(steps, np.int32)
        original = arrays[x_index] + arrays[weight_index]
        first, distance = update(original, arrays[weight_index], arrays[initial_index], steps, 80, -3, 1)
        follow, follow_distance = update(original, arrays[weight_index], first, steps, 31, -2, -1)
        expected = [follow, follow_distance, first, distance, original, arrays[initial_index], arrays[step_index]]
        caller = tmp_path / f"caller-{index}.npz"
        np.savez(caller, **{f"in{i}": value for i, value in enumerate(arrays)})
        actual, report = invoke_merlin_host(out / "consumer", caller, tmp_path / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["functional_inputs_preserved"]
        for value, reference in zip(actual, expected, strict=True):
            assert (value.shape, value.dtype, value.tobytes()) == (reference.shape, reference.dtype, reference.tobytes())
        assert _record(out / "consumer/host/model.so") == library
    for index, steps in enumerate((-1, 5)):
        arrays[step_index] = np.asarray(steps, np.int32)
        caller = tmp_path / f"bad-caller-{index}.npz"
        np.savez(caller, **{f"in{i}": value for i, value in enumerate(arrays)})
        refused = tmp_path / f"guard-refusal-{index}"
        report = observe_merlin_host(out / "consumer", caller, refused,
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["returncode"] == -signal.SIGABRT
        assert b"runtime scalar outside declared bounds: steps" in (
            (refused / "stdout.bin").read_bytes() + (refused / "stderr.bin").read_bytes())
        assert not (refused / "outputs.npz").exists() and not (refused / "evaluation.json").exists()
        assert _record(out / "consumer/host/model.so") == library
    options = {"merlin_root": os.environ["T2M_MERLIN_ROOT"], "python": os.environ["T2M_MERLIN_PYTHON"]}
    report = target.qualify(out, tmp_path / "target", comparison="words", **options)
    assert target.verify(tmp_path / "target", **options) == report and report["evaluation"]["result_count"] == 7
