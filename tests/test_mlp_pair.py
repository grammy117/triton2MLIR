"""Original model2MLIR plain/Triton workloads, through the same consumer ABI."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
M2M, ROOT = os.environ.get("T2M_MODEL2MLIR_PYTHON"), os.environ.get("T2M_MODEL2MLIR_ROOT")
pytestmark = pytest.mark.skipif(not all((OPT, NATIVE, M2M, ROOT)), reason="existing compiler/native/model2MLIR source environments required")


def test_existing_plain_and_triton_mlp_have_the_same_complete_native_result(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/capture_mlp_pair.py"
    process = subprocess.run([M2M, str(script), "--model2mlir-root", ROOT, "--out", str(tmp_path / "captures")],
                             capture_output=True, text=True, timeout=60,
                             env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(tmp_path)))
    assert process.returncode == 0, process.stderr
    plain, triton = tmp_path / "captures/plain", tmp_path / "captures/triton"
    np.testing.assert_array_equal(np.load(plain / "golden.npy"), np.load(triton / "golden.npy"))
    assert (plain / "weights.safetensors").read_bytes() == (triton / "weights.safetensors").read_bytes()

    # Reuse the original JIT function unchanged, isolating its two Triton
    # imports from the independently pinned Torch/model2MLIR environment.
    original = Path(ROOT) / "workloads/mlp_basic_triton/kernels.py"
    node = next(n for n in ast.parse(original.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_mlp2_kernel")
    extracted = tmp_path / "original_jit_function.py"
    extracted.write_text(f"# Original source: {original}\nimport triton\nimport triton.language as tl\n" + ast.unparse(node) + "\n")
    spec = importlib.util.spec_from_file_location("original_mlp_jit_fixture", extracted)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = ("x_ptr", "w1_ptr", "b1_ptr", "w2_ptr", "b2_ptr", "out_ptr")
    shapes = ((4, 16), (32, 16), (32,), (32, 32), (32,), (4, 32))
    constants = {"K": 16, "H1": 32, "H2": 32, "BLOCK_M": 16}
    source = tmp_path / "kernel.ttir"
    source.write_text(capture_ttir(module._mlp2_kernel, {**{n: "*fp32" for n in names}, "M": "i32",
                                                      **{n: "constexpr" for n in constants}}, constants))
    abi = TensorABI("_mlp2_kernel", tuple(TensorBuffer(n, s, "write" if n == "out_ptr" else "read")
                    for n, s in zip(names, shapes, strict=True)), scalar_constants=(4,))
    kernel = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT)
    assert sum(op.name == "linalg.matmul" for op in kernel.module.walk()) == 2
    assert any(op.name == "tensor.pad" for op in kernel.module.walk())
    binding = KernelBinding("triton_exp::fused_mlp2", kernel,
                            operands=dict(zip(names[:-1], ("x", "w1", "b1", "w2", "b2"), strict=True)))
    for name, capture, bindings in (("plain", plain, []), ("triton", triton, [binding])):
        write_model_bundle(capture, bindings, tmp_path / (name + "-bound"), work=tmp_path / (name + "-work"),
                           python=NATIVE, merlin_root=os.environ.get("T2M_MERLIN_ROOT"),
                           merlin_python=os.environ.get("T2M_MERLIN_PYTHON"), rtol=1e-5, atol=1e-6)
