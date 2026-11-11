"""Source-specialized packed models retain math, custody and complete results."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_grouped_maps import capture as translate
from test_spike_model import Q, TOOLS as TARGET_TOOLS

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


@pytest.mark.skipif(not TOOLS, reason="existing model2MLIR, native and Merlin tools required")
@pytest.mark.parametrize("dtype,warps,control", [("i64", 2, False), ("f32", 4, False), ("i64", 4, True)])
def test_source_specialized_model_preserves_grouping_and_all_results(tmp_path, dtype, warps, control):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding, bind_model
    from t2m.model_bundle import write_model_bundle
    from t2m.model_source_layout import PROFILE, validate_model
    from t2m.model_tensor_views import POLICY
    from t2m.shared_tensor import convert_shared, parse_shared

    source = tmp_path / "source"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_grouped_map_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(script), "--out", str(source), "--dtype", dtype,
        "--warps", str(warps), *(["--control"] if control else [])], capture_output=True, text=True, timeout=60,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stdout + process.stderr
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    result, _ = translate(kernel, 2, dtype, False, warps, "guarded-buffer")
    if control:
        from t2m.examples.grouped_map_kernels import grouped_control_map
        from t2m.frontend import capture_ttir
        path = kernel / "control.ttir"
        path.write_text(capture_ttir(grouped_control_map, {**{n: "*i64" for n in ("x", "y", "out_mix", "out_asymmetric")},
            "n": "i32", "B": "constexpr"}, {"B": 256}))
        result = convert_shared(path, replace(result.spec, entry="grouped_control_map"), kernel / "control-translation",
            optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile="guarded-buffer", address_policy="source-words",
            source_layout=result.trace["source_layout_specialization"]["source_options"])
    mapping = {"target": "source_target", "num_warps": "source_warps", "threads_per_warp": "source_threads", "num_ctas": "source_ctas"}
    scalars = {"n": 300, "source_target": "cuda:80", "source_warps": warps, "source_threads": 32, "source_ctas": 1, "control": control}
    binding = KernelBinding("t2m_demo::source_grouped_map", result, tensor_views=POLICY,
        scalar_arguments=scalars, runtime_arguments={"n": "n"}, source_layout_arguments=mapping)
    parent = (source / "model.mlir").read_text()
    with pytest.raises(ValueError, match="every source layout option"):
        bind_model(parent, [replace(binding, source_layout_arguments={})])
    with pytest.raises(ValueError, match="disagrees with captured scalar"):
        bind_model(parent, [replace(binding, scalar_arguments={**scalars, "source_warps": 8})])
    out = tmp_path / "bound"
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    write_model_bundle(source, [binding], out, work=tmp_path / "host", python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=root, merlin_python=python, merlin_host=True, merlin_clang=clang)
    verify_bundle(out)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (source / name).read_bytes()
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert trace["source_layout_binding_profile"] == PROFILE
    assert len(trace["calls"]) == 2 and len(trace["kernels"]) == 1 and trace["model_abi"] == "unchanged"
    assert all(set(call["source_layout_bindings"]) == set(mapping) for call in trace["calls"])
    for damage in ("profile", "selection", "attribute", "missing"):
        broken = json.loads(json.dumps(trace))
        if damage == "profile":
            broken.pop("source_layout_binding_profile")
        elif damage == "selection":
            next(iter(broken["kernels"].values()))["source_layout_specialization"]["source_options"]["num_warps"] = 8
        elif damage == "attribute":
            broken["calls"][0]["source_layout_bindings"]["num_warps"]["attribute"] = "8 : i64"
        else:
            broken["calls"][0].pop("source_layout_bindings")
        with pytest.raises(ValueError):
            validate_model(broken, parse_shared(parent))
    # Rebinding an outer artifact digest cannot replace source-call custody.
    binding_path, receipt_path = out / "kernel-bindings.json", out / "capture_receipt.json"
    original_binding, original_receipt = binding_path.read_bytes(), receipt_path.read_bytes()
    broken = json.loads(original_binding)
    broken["calls"][0].pop("source_layout_bindings")
    try:
        from t2m.bundle import _record
        binding_path.write_text(json.dumps(broken))
        receipt = json.loads(original_receipt)
        receipt["artifacts"]["kernel-bindings.json"] = _record(binding_path)
        receipt_path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="every source layout option"):
            verify_bundle(out)
    finally:
        binding_path.write_bytes(original_binding)
        receipt_path.write_bytes(original_receipt)
    with np.load(source / "goldens.npz", allow_pickle=False) as expected:
        for path in (tmp_path / "host/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                assert actual.files == [f"out{i}" for i in range(4)]
                for key in actual.files:
                    assert (actual[key].shape, actual[key].dtype, actual[key].tobytes()) == (
                        expected[key].shape, expected[key].dtype, expected[key].tobytes())
    if TARGET_TOOLS:
        report = Q.qualify(out, tmp_path / "target", merlin_root=root, python=python, comparison="words")
        assert report["evaluation"]["result_count"] == 4
        assert Q.verify(tmp_path / "target", merlin_root=root, python=python) == report
