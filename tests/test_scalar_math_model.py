"""Real model2MLIR math capture keeps weights and all eight model results."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from t2m.bundle import verify_bundle
from t2m.frontend import capture_ttir
from t2m.model_binding import KernelBinding
from t2m.model_bundle import write_model_bundle
from t2m.shared_tensor import TensorABI,TensorBuffer,convert_shared

REQUIRED = all(os.environ.get(key) for key in ("TRITON_SHARED_OPT","T2M_NATIVE_PYTHON","T2M_MODEL2MLIR_PYTHON",
    "T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED,reason="existing model2MLIR/native/Merlin environments required")


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    root = tmp_path_factory.mktemp("scalar-math-model")
    source = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_scalar_math_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"],str(script),"--out",str(source)],capture_output=True,text=True,
        timeout=60,env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TMPDIR=str(root)))
    assert process.returncode == 0,process.stdout+process.stderr
    return source


@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
def test_precise_math_model_reaches_the_actual_merlin_interface(tmp_path,captured,profile):
    from t2m.examples.scalar_math_kernels import precise_values
    names = ["x","y","low","high","no_nan","all_nan","divided","root"]
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(precise_values,{**{name:"*fp32" for name in names},"n":"i32","B":"constexpr"},{"B":8}))
    abi = TensorABI(precise_values.__name__,tuple(TensorBuffer(name,(97,),"read" if i < 4 else "write") for i,name in enumerate(names)),
        grid=(13,),scalar_constants=(97,))
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=os.environ["TRITON_SHARED_OPT"],memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    binding = KernelBinding("t2m_demo::precise_values",result,operands={name:name for name in names[:4]},scalar_arguments={"n":97})
    out = tmp_path / "model"
    write_model_bundle(captured,[binding],out,work=tmp_path / "qualification",python=os.environ["T2M_NATIVE_PYTHON"],
        merlin_root=os.environ["T2M_MERLIN_ROOT"],merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_host=True,merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(out)
    for name in ("weights.safetensors","weights.safetensors.manifest.json","inputs.npz","extra.npz","golden.npy","goldens.npz"):
        assert (captured / name).read_bytes() == (out / name).read_bytes()
    host = json.loads((out / "consumer/host/execution.json").read_text())
    assert host["host_execution"] == host["reference_comparison"] == "passed" and host["result_count"] == 8
    assert "math.sqrt" in (out / "model.mlir").read_text() and "arith.divf" in (out / "model.mlir").read_text()
