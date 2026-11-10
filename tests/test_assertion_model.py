"""A real checked Triton model keeps weights, scalar bindings and all results."""

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

REQUIRED = all(os.environ.get(k) for k in ("TRITON_SHARED_OPT","T2M_NATIVE_PYTHON","T2M_MODEL2MLIR_PYTHON",
    "T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED,reason="existing model2MLIR/native/Merlin toolchains required")


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    root = tmp_path_factory.mktemp("assertion-model")
    source = root / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_assertion_model.py"
    process = subprocess.run([os.environ["T2M_MODEL2MLIR_PYTHON"],str(script),"--out",str(source)],capture_output=True,text=True,timeout=60,
        env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TMPDIR=str(root)))
    assert process.returncode == 0, process.stdout + process.stderr
    return source


@pytest.mark.parametrize("profile", ["tensor","buffer","guarded-buffer"])
def test_source_assertions_use_the_complete_model2mlir_interface(tmp_path,captured,profile):
    from t2m.examples.assertion_kernels import checked_copy
    ttir = tmp_path / "source.ttir"
    ttir.write_text(capture_ttir(checked_copy,{"x":"*i32","out":"*i32","n":"i32","steps":"i32","limit":"i32",
        "B":"constexpr","MODE":"constexpr","RANK":"constexpr"},{"B":8,"MODE":2,"RANK":2},debug=True))
    abi = TensorABI("checked_copy",(TensorBuffer("x",(17,),dtype="i32"),TensorBuffer("out",(17,),"readwrite","i32")),
        scalar_constants=(13,3,5),grid=(3,))
    result = convert_shared(ttir,abi,tmp_path / "translation",optimizer=os.environ["TRITON_SHARED_OPT"],memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    binding = KernelBinding("t2m_demo::checked_copy",result,operands={"x":"x","out":"initial"},scalar_arguments={"n":13,"steps":3,"limit":5})
    out, work = tmp_path / "model", tmp_path / "qualification"
    write_model_bundle(captured,[binding],out,work=work,python=os.environ["T2M_NATIVE_PYTHON"],merlin_root=os.environ["T2M_MERLIN_ROOT"],
        merlin_python=os.environ["T2M_MERLIN_PYTHON"],merlin_host=True,merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(out)
    for name in ("weights.safetensors","weights.safetensors.manifest.json","inputs.npz","extra.npz","golden.npy","goldens.npz"):
        assert (out / name).read_bytes() == (captured / name).read_bytes()
    assert "cf.assert" in (out / "model.mlir").read_text()
    compatibility = json.loads((out / "compatibility.json").read_text())
    assert any("source assertions" in item for item in compatibility["kernel_profiles"]["kernels/0"]["differences"])
    host = json.loads((out / "consumer/host/execution.json").read_text())
    assert host["host_execution"] == host["reference_comparison"] == "passed"
    assert host["result_count"] == 3
    with np.load(captured / "goldens.npz",allow_pickle=False) as expected:
        for filename in (work / "native/outputs.npz",out / "consumer/host/outputs.npz"):
            with np.load(filename,allow_pickle=False) as actual:
                for i in range(3):
                    np.testing.assert_array_equal(actual[f"out{i}"],expected[f"out{i}"])
