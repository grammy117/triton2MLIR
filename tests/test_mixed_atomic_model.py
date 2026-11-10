"""Mixed atomic kernels bind real model2MLIR weights and all three results."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

OPT, NATIVE, M2M = (os.environ.get(n) for n in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MODEL2MLIR_PYTHON"))
MERLIN, CLANG = os.environ.get("T2M_MERLIN_ROOT"), os.environ.get("T2M_MERLIN_CLANG")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and M2M and MERLIN and CLANG), reason="existing shared/model2MLIR/Merlin required")


@pytest.mark.parametrize("width,kind", [(16, "cas"), (32, "cas"), (64, "cas"), (32, "rmw"), (64, "rmw")])
def test_model_keeps_both_storage_dtypes_weights_and_every_tuple_reference(tmp_path, width, kind):
    from t2m.bundle import verify_bundle
    from t2m.examples.atomic_kernels import selected_word_cas, selected_word_rmw
    from t2m.frontend import capture_ttir
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    capture = tmp_path / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_mixed_atomic_model.py"
    process = subprocess.run([M2M, str(script), "--out", str(capture), "--width", str(width), "--kind", kind],
        capture_output=True, text=True, timeout=60, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr
    source = tmp_path / "source.ttir"
    common = {"B": 8, "WIDTH": width, "SCALAR": False}
    if kind == "cas":
        kernel, capacity, constants = selected_word_cas, 17, {**common, "LANE": True, "choice": False}
        names, types = ("comparison", "value"), (f"i{width}", f"i{width}")
        operands = {"a": "a", "b": "b", "comparison": "index", "value": "x"}
    else:
        kernel, capacity, constants = selected_word_rmw, 37, {**common, "n": 16, "choice": True, "OP": "xor"}
        names, types = ("value", "index"), (f"i{width}", "i32")
        operands = {"a": "a", "b": "b", "value": "x", "index": "index"}
    source.write_text(capture_ttir(kernel, {"a": f"*fp{width}", "b": f"*i{width}", "old": f"*fp{width}",
        **{name: "*" + dtype for name, dtype in zip(names, types)}, **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI(kernel.__name__, (TensorBuffer("a", (capacity,), "readwrite", dtype=f"f{width}"),
        TensorBuffer("b", (capacity,), "readwrite", dtype=f"i{width}"),
        *(TensorBuffer(name, (16,), dtype=dtype) for name, dtype in zip(names, types)),
        TensorBuffer("old", (16,), "write", dtype=f"f{width}")), grid=(2,))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    binding = KernelBinding("t2m_demo::mixed_atomic_update", result, operands=operands)
    out = tmp_path / "model"
    write_model_bundle(capture, [binding], out, work=tmp_path / "qualification", python=NATIVE,
        merlin_root=MERLIN, merlin_python=os.environ.get("T2M_MERLIN_PYTHON"), merlin_host=True, merlin_clang=CLANG)
    receipt = verify_bundle(out)
    assert receipt["schema"] == "t2m.bound-model-receipt.v1"
    assert json.loads((out / "execution.json").read_text())["result_count"] == 3
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "golden.npy", "goldens.npz", "extra.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    with np.load(capture / "goldens.npz", allow_pickle=False) as expected:
        for path in (tmp_path / "qualification/native/outputs.npz", out / "consumer/host/outputs.npz"):
            with np.load(path, allow_pickle=False) as actual:
                for i in range(3):
                    np.testing.assert_array_equal(actual[f"out{i}"].view(getattr(np, f"uint{width}")), expected[f"out{i}"].view(getattr(np, f"uint{width}")))
    host = json.loads((out / "consumer/host/execution.json").read_text())
    assert host["host_execution"] == host["reference_comparison"] == "passed" and host["result_count"] == 3
    assert host["evaluation"]["functional_inputs_preserved"]
