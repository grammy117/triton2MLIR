"""Full model2MLIR packages retain source assembly, learned words and all results."""

import importlib.util
import json
import os
import subprocess
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import pytest

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
PROVIDER = bool(entry_points(group="triton2mlir.source_assembly.v1", name="ptx-integer-v1"))
pytestmark = pytest.mark.skipif(not (TOOLS and PROVIDER), reason="optional provider, model2MLIR and native/Merlin tools required")


@pytest.mark.parametrize("dtype", ("i64", "i32"))
def test_learned_models_preserve_source_programs_and_all_six_results(tmp_path, dtype):
    from t2m.bundle import verify_bundle
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.assembly_kernels import assembly_word_state
    from t2m.frontend import capture_ttir
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("assembly_model_sample", root / "adapters/ptx/examples/model_reference.py")
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    capture = tmp_path / "capture"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"], str(root / "scripts/capture_assembly_model.py"),
        "--out", str(capture), "--dtype", dtype], capture_output=True, text=True, timeout=60, check=False,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stdout + process.stderr
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    width = 64 if dtype == "i64" else 32
    names = ("x", "y", "shifts", "out_a", "out_b")
    source = kernel / "source.ttir"
    source.write_text(capture_ttir(assembly_word_state, {**{name: "*i32" if name == "shifts" else "*" + dtype for name in names},
        "n": "i32", "iterations": "i32", "ASM": "constexpr", "B": "constexpr", "WORDS64": "constexpr"},
        {"ASM": sample.program(width), "B": 32, "WORDS64": width == 64}))
    abi = TensorABI("assembly_word_state", tuple(TensorBuffer(name, ("n",), "write" if name.startswith("out") else "read",
        "i32" if name == "shifts" else dtype) for name in names), grid=(GridExtent("n", 32),),
        runtime_scalars=(RuntimeScalar("n", "i32", 5, bounds=(0, 67)), RuntimeScalar("iterations", "i32", 6, bounds=(0, 4))))
    result = convert_shared(source, abi, kernel / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer", address_policy="source-words", assembly_provider="ptx-integer-v1")
    operands = {name: name for name in ("x", "y", "shifts")}
    bindings = [KernelBinding("t2m_demo::assembly_word_state", result, tensor_views=POLICY, operands=operands,
        scalar_arguments={"n": n, "iterations": iterations}, runtime_arguments={"n": "n", "iterations": "iterations"})
        for n, iterations in ((32, 3), (32, 2), (0, 4))]
    out = tmp_path / "bound"
    write_model_bundle(capture, bindings, out, work=tmp_path / "qualification", python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"], merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_host=True, merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(out)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert len(trace["calls"]) == 3 and len(trace["kernels"]) == 1 and trace["model_abi"] == "unchanged"
    for path in (tmp_path / "qualification/native/outputs.npz", out / "consumer/host/outputs.npz"):
        with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(path, allow_pickle=False) as actual:
            assert actual.files == [f"out{i}" for i in range(6)]
            for name in actual.files:
                assert actual[name].shape == expected[name].shape and actual[name].dtype == expected[name].dtype
                assert actual[name].tobytes() == expected[name].tobytes()
    if dtype == "i64":
        from t2m.cli import main

        configuration = {"schema": "t2m.kernel-bindings.v1", "kernels": [{
            "kernel": binding.qualname, "ttir": "kernel/source.ttir", "abi": abi.as_dict(), "operands": operands,
            "tensor_views": POLICY, "scalar_arguments": binding.scalar_arguments, "runtime_arguments": binding.runtime_arguments,
            "memory_profile": "guarded-buffer", "address_policy": "source-words",
            **({"assembly_provider": "ptx-integer-v1"} if index == 0 else {}),
        } for index, binding in enumerate(bindings)]}
        path = tmp_path / "bindings.json"
        path.write_text(json.dumps(configuration))
        cli = tmp_path / "cli-bound"
        assert main(["bind-model", str(capture), "--bindings", str(path), "--out", str(cli),
            "--assembly-provider", "ptx-integer-v1", "--optimizer", os.environ["TRITON_SHARED_OPT"],
            "--python", os.environ["T2M_NATIVE_PYTHON"], "--merlin-root", os.environ["T2M_MERLIN_ROOT"],
            "--merlin-python", os.environ["T2M_MERLIN_PYTHON"], "--merlin-clang", os.environ["T2M_MERLIN_CLANG"], "--merlin-host"]) == 0
        verify_bundle(cli)
        with np.load(capture / "goldens.npz", allow_pickle=False) as expected, np.load(cli / "consumer/host/outputs.npz", allow_pickle=False) as actual:
            assert actual.files == expected.files
            for name in actual.files:
                assert actual[name].shape == expected[name].shape and actual[name].dtype == expected[name].dtype
                assert actual[name].tobytes() == expected[name].tobytes()
