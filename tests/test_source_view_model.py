"""Source allocation roots replace only independently recovered model views."""

import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from test_spike_model import Q, TOOLS as TARGET_TOOLS

OPT, NATIVE, M2M = (os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON"))
TOOLS = OPT and NATIVE and M2M and all(os.environ.get(name) for name in
    ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


def implementation(root, dtype="i64"):
    from t2m.allocation_bindings import PointerArgument, ScalarArgument, convert_allocations
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.model_view_kernels import update_source_view
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer
    source = root / "source.ttir"
    source_dtype = "fp32" if dtype == "f32" else dtype
    source.write_text(capture_ttir(update_source_view, {"owner": "*" + source_dtype, "view": "*" + source_dtype, "out": "*" + source_dtype,
        **{name: "i32" for name in ("rows", "cols", "s0", "s1", "steps")}, "bias": "i64", "BM": "constexpr", "BN": "constexpr"},
        {"BM": 8, "BN": 4}))
    names = ("rows", "cols", "s0", "s1", "origin", "capacity", "steps", "bias", "count")
    bounds = ((0, 16), (0, 16), (1, 64), (1, 64), (0, 96), (0, 96), (0, 3), (-9, 9), (0, 96))
    scalars = tuple(RuntimeScalar(name, "i64" if name in ("origin", "bias") else "i32", i + 2, bounds=bound)
        for i, (name, bound) in enumerate(zip(names, bounds, strict=True)))
    abi = TensorABI("source_view_root", (TensorBuffer("state", ("capacity",), "readwrite", dtype),
        TensorBuffer("out", ("count",), "write", dtype)), runtime_scalars=scalars,
        grid=(GridExtent("rows", 8), GridExtent("cols", 4)), intent="Source model views share one complete owning allocation")
    arguments = (PointerArgument("state"), PointerArgument("state", "origin"), PointerArgument("out"),
        *(ScalarArgument(name) for name in ("rows", "cols", "s0", "s1", "steps", "bias")))
    return convert_allocations(source, abi, arguments, root / "translation", source_entry="update_source_view",
        optimizer=OPT, memory_profile="guarded-buffer", address_policy="source-words")


def bindings(result):
    from t2m.model_binding import KernelBinding
    names = ("rows", "cols", "s0", "s1", "origin", "capacity", "steps", "bias", "count")
    cases = ((8, 6, 2, 16, 1, 96, 2, 5, 48), (3, 5, 32, 3, 2, 96, 1, -3, 15), (0, 6, 1, 16, 0, 96, 3, 2, 0))
    return [KernelBinding("t2m_demo::source_view_update", result, operands={"state": "owner"},
        scalar_arguments=dict(zip(names, values, strict=True)), runtime_arguments={name: name for name in names},
        tensor_views="dense-row-major-v1", source_views={"view": {"root": "state", "source_arg": 1,
            "shape": ["rows", "cols"], "strides": ["s0", "s1"], "offset": "origin"}}) for values in cases]


@pytest.mark.skipif(not TOOLS, reason="existing source/model2MLIR/Merlin tools required")
@pytest.mark.parametrize("dtype", ("i64", "f32", "bf16"))
def test_actual_non_dense_model_views_keep_state_snapshots_and_complete_results(tmp_path, dtype):
    from t2m.bundle import verify_bundle
    from t2m.model_bundle import write_model_bundle, _parent, _references
    from t2m.shared_tensor import parse_shared
    from xdsl.dialects import func
    capture = tmp_path / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_source_view_model.py"
    process = subprocess.run([M2M, str(script), "--out", str(capture), "--dtype", dtype], capture_output=True,
        timeout=60, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr.decode()
    result = implementation(tmp_path, dtype)
    out = tmp_path / "bound"
    selected = bindings(result)
    unrelated = replace(selected[-1], qualname="other::source_view_update", operands={"state": "foreign"})
    write_model_bundle(capture, [*selected, unrelated], out, work=tmp_path / "execution", python=NATIVE,
        merlin_root=os.environ["T2M_MERLIN_ROOT"], merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_clang=os.environ["T2M_MERLIN_CLANG"], merlin_host=True)
    verify_bundle(out)
    check_structure(out)
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert trace["schema"] == "t2m.model-kernel-binding.v3"
    assert len(trace["calls"]) == 3 and len(trace["kernels"]) == 1
    layouts = [call["source_tensor_views"]["view"]["layout"] for call in trace["calls"]]
    assert [(view["shape"], view["strides"], view["offset"]) for view in layouts] == [
        ([8, 6], [2, 16], 1), ([3, 5], [32, 3], 2), ([0, 6], [1, 16], 0)]
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "goldens.npz", "extra.npz"):
        assert (capture / name).read_bytes() == (out / name).read_bytes()
    parent = _parent(capture)
    entry = next(op for op in parse_shared((capture / "model.mlir").read_text()).ops
                 if isinstance(op, func.FuncOp) and op.sym_name.data == "forward")
    expected, _ = _references(capture, parent, entry.function_type.outputs)
    assert len(expected) == 7 and expected[5].shape == (0, 6)
    for path in (tmp_path / "execution/native/outputs.npz", out / "consumer/host/outputs.npz"):
        with np.load(path, allow_pickle=False) as actual:
            assert actual.files == [f"out{i}" for i in range(7)]
            for i, gold in enumerate(expected):
                value = actual[f"out{i}"]
                assert (value.shape, value.dtype, value.tobytes()) == (gold.shape, gold.dtype, gold.tobytes())
    assert any("t2m.model-kernel-binding.v3" in line for line in json.loads((out / "compatibility.json").read_text())["differences"])
    if TARGET_TOOLS:
        report = Q.qualify(out, tmp_path / "target", merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"], comparison="words")
        assert report["evaluation"]["result_count"] == 7
        assert Q.verify(tmp_path / "target", merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"]) == report
    if dtype == "i64":
        check_cli(tmp_path, capture, result)
        check_rehashed_refusals(out, tmp_path)


def check_structure(out):
    from t2m.shared_tensor import parse_shared
    module = parse_shared((out / "model.mlir").read_text())
    grids = [op for op in module.walk() if op.name == "scf.forall"]
    assert len(grids) == 3
    for grid in grids:
        assert any(op.name == "scf.for" for op in grid.walk())
        assert "mapping" not in grid.attributes and "mapping" not in grid.properties
        assert all(name in grid.attributes for name in ("prov.family", "prov.op", "prov.region_id", "prov.emitted_id",
            "prov.source_accesses", "prov.kernel_source_view_profile", "prov.kernel_view_arguments"))
    assert all(not op.name.startswith(("tt.", "ttg.", "gpu.", "nvvm.")) for op in module.walk())


def check_cli(root, capture, result):
    config = {"schema": "t2m.kernel-bindings.v1", "kernels": [{"kernel": binding.qualname,
        "ttir": "source.ttir", "abi": result.spec.as_dict(),
        "allocation_bindings": json.loads(result.auxiliary_artifacts["allocation-arguments.json"]),
        "operands": binding.operands, "runtime_arguments": binding.runtime_arguments,
        "scalar_arguments": binding.scalar_arguments, "tensor_views": binding.tensor_views,
        "source_views": binding.source_views, "memory_profile": "guarded-buffer", "address_policy": "source-words"}
        for binding in bindings(result)]}
    path = root / "bindings.json"
    path.write_text(json.dumps(config))
    process = subprocess.run([sys.executable, "-m", "t2m", "bind-model", str(capture), "--bindings", str(path),
        "--out", str(root / "cli"), "--optimizer", OPT, "--python", NATIVE,
        "--merlin-root", os.environ["T2M_MERLIN_ROOT"], "--merlin-python", os.environ["T2M_MERLIN_PYTHON"],
        "--merlin-host", "--merlin-clang", os.environ["T2M_MERLIN_CLANG"]], capture_output=True, timeout=120,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(root)))
    assert process.returncode == 0, process.stderr.decode()
    from t2m.bundle import verify_bundle
    verify_bundle(root / "cli")


def check_rehashed_refusals(out, root):
    from t2m.bundle import verify_bundle, _record
    trace_path, receipt_path = out / "kernel-bindings.json", out / "capture_receipt.json"
    original_trace, original_receipt = trace_path.read_bytes(), receipt_path.read_bytes()
    worker = '''import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]+'/src')
from merlin.frontends.compile_inputs import stage_compile_inputs
read=Path.read_bytes
def checked(path):
    if path.name in ('inputs.npz','golden.npy','goldens.npz','extra.npz','tensor_words.npz','execution.json'):
        raise AssertionError('compiler read evaluator data')
    return read(path)
Path.read_bytes=checked
try: stage_compile_inputs(sys.argv[2],sys.argv[3])
except ValueError as error:
    assert 'view' in str(error) or str(error)=='derived runtime kernel binding disagrees with its original typed call',str(error)
else: raise AssertionError('forged source view accepted')
'''
    try:
        for change in ("layout", "source-pointer", "downgrade", "profile", "missing-call"):
            trace, receipt = json.loads(original_trace), json.loads(original_receipt)
            view = trace["calls"][0]["source_tensor_views"]["view"]
            if change == "layout": view["layout"]["offset"] = 0
            elif change == "source-pointer": view["binding"]["source_arg"] = 0
            elif change == "downgrade": trace["schema"] = "t2m.model-kernel-binding.v2"
            elif change == "profile": trace.pop("source_tensor_view_binding_profile")
            else: trace["calls"][0].pop("source_tensor_views")
            trace_path.write_text(json.dumps(trace))
            receipt["artifacts"]["kernel-bindings.json"] = _record(trace_path)
            receipt_path.write_text(json.dumps(receipt))
            with pytest.raises(ValueError): verify_bundle(out)
            target = root / ("rejected-" + change)
            process = subprocess.run([os.environ["T2M_MERLIN_PYTHON"], "-c", worker,
                os.environ["T2M_MERLIN_ROOT"], str(out), str(target)], capture_output=True, timeout=30)
            assert process.returncode == 0, process.stderr.decode()
            assert not target.exists()
    finally:
        trace_path.write_bytes(original_trace)
        receipt_path.write_bytes(original_receipt)
