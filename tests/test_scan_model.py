import json
import os
from pathlib import Path
import subprocess

import pytest

from t2m.bundle import verify_bundle
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
M2M, MERLIN = os.environ.get("T2M_MODEL2MLIR_PYTHON"), os.environ.get("T2M_MERLIN_ROOT")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and M2M), reason="existing shared/native/model2MLIR environments required")


@pytest.mark.parametrize("kind", ["sum", "product", "tuple"])
def test_actual_scan_model_preserves_kernel_results_weights_and_merlin_interface(tmp_path, kind):
    script = Path(__file__).resolve().parents[1] / "scripts/capture_scan_model.py"
    capture = tmp_path / "capture"
    process = subprocess.run([M2M, str(script), "--out", str(capture), "--kind", kind, "--reverse"],
                             capture_output=True, text=True, timeout=60,
                             env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr
    from t2m.examples.tensor_kernels import scan_tiles, tuple_scan
    if kind == "tuple":
        kernel, constants, grid = tuple_scan, {"R": 2, "S": 8, "AXIS": 1, "REVERSE": True}, (1,)
        buffers = tuple(TensorBuffer(n, (2, 8), "write" if n.startswith("out") else "read")
                        for n in ("a", "b", "out_a", "out_b"))
        identity, scalar = "t2m_demo::affine_scan", {"reverse": True}
    else:
        kernel, constants, grid = scan_tiles, {"N": 53, "B": 16, "KIND": 0 if kind == "sum" else 1, "REVERSE": True}, (4,)
        buffers = (TensorBuffer("x", (53,)), TensorBuffer("out", (53,), "write"))
        identity, scalar = "t2m_demo::scan", {"reverse": True, "kind": constants["KIND"]}
    source = tmp_path / "kernel.ttir"
    source.write_text(capture_ttir(kernel, {**{b.name: "*fp32" for b in buffers},
                                          **{n: "constexpr" for n in constants}}, constants))
    implementation = convert_shared(source, TensorABI(kernel.__name__, buffers, grid=grid), tmp_path / "translation", optimizer=OPT)
    out = tmp_path / "model"
    write_model_bundle(capture, [KernelBinding(identity, implementation, scalar_arguments=scalar)],
                       out, work=tmp_path / "qualification", python=NATIVE,
                       merlin_root=MERLIN, merlin_python=os.environ.get("T2M_MERLIN_PYTHON"), rtol=1e-5, atol=3e-6)
    assert verify_bundle(out)["schema"] == "t2m.bound-model-receipt.v1"
    for name in ("golden.npy", "weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    bindings = json.loads((out / "kernel-bindings.json").read_text())
    assert len(bindings["calls"][0]["source_result_order"]) == (2 if kind == "tuple" else 1)
    if MERLIN:
        consumer = json.loads((out / "consumer/interface.json").read_text())
        assert consumer["interface_ingestion"] == consumer["runtime_generation"] == "passed"
        assert consumer["target_execution"] == "not-performed"
