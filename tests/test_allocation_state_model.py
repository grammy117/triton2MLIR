"""Actual parent calls retain root updates, learned state and every result."""

import json
import os
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

OPT,NATIVE,M2M = (os.environ.get(name) for name in ("TRITON_SHARED_OPT","T2M_NATIVE_PYTHON","T2M_MODEL2MLIR_PYTHON"))
REQUIRED = OPT and NATIVE and M2M and all(os.environ.get(name) for name in
    ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED,reason="existing shared/model2MLIR/Merlin environments required")


def implementation(root):
    from t2m.allocation_bindings import (
        PointerArgument,
        ScalarArgument,
        convert_allocations,
    )
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.readonly_alias_kernels import aliased_updates
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer
    source = root / "source.ttir"
    source.write_text(capture_ttir(aliased_updates,{"read":"*i64","first":"*i64","second":"*i64",
        "n":"i32","steps":"i32","B":"constexpr"},{"B":8}))
    scalars = (RuntimeScalar("n","i32",1,bounds=(0,17)),RuntimeScalar("steps","i32",2,bounds=(0,3)),
        *(RuntimeScalar(name,"i64",index,bounds=(-32,63)) for index,name in enumerate(("read","first","second"),3)),
        RuntimeScalar("capacity","i32",6,bounds=(0,48)))
    abi = TensorABI("model_allocation_state",(TensorBuffer("state",("capacity",),"readwrite","i64"),),
        runtime_scalars=scalars,grid=(GridExtent("n",8),),intent="Shared source root updates within original model tensor values")
    bindings = (*(PointerArgument("state",name) for name in ("read","first","second")),ScalarArgument("n"),ScalarArgument("steps"))
    return convert_allocations(source,abi,bindings,root / "translation",source_entry="aliased_updates",optimizer=OPT,
        memory_profile="guarded-buffer",address_policy="source-words")


def test_real_model_dense_views_bind_shared_root_updates_and_keep_all_references(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding, bind_model
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY
    capture = tmp_path / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_allocation_state_model.py"
    process = subprocess.run([M2M,str(script),"--out",str(capture)],capture_output=True,text=True,timeout=60,check=False,
        env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TMPDIR=str(tmp_path)))
    assert process.returncode == 0,process.stderr
    result = implementation(tmp_path)
    bindings = [KernelBinding("t2m_demo::allocation_state",result,tensor_views=POLICY,
        scalar_arguments=dict(zip(("n","steps","read_origin","first_origin","second_origin","capacity"),values,strict=True)),
        runtime_arguments={"n":"n","steps":"steps","read":"read_origin","first":"first_origin","second":"second_origin","capacity":"capacity"})
        for values in ((17,2,0,0,0,48),(7,2,4,0,8,48),(0,2,0,0,0,0))]
    with pytest.raises(ValueError,match="matching typed/scalar"):
        bind_model((capture / "model.mlir").read_text(),[replace(binding,tensor_views=None) for binding in bindings])
    with pytest.raises(ValueError,match="unknown model tensor view"):
        bind_model((capture / "model.mlir").read_text(),[replace(bindings[0],tensor_views="unknown")])
    out = tmp_path / "bound"
    write_model_bundle(capture,bindings,out,work=tmp_path / "qualification",python=NATIVE,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_host=True,merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(out)
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert trace["schema"] == "t2m.model-kernel-binding.v2" and len(trace["calls"]) == 3
    assert len(trace["kernels"]) == 1 and trace["model_abi"] == "unchanged"
    for name in ("weights.safetensors","weights.safetensors.manifest.json","inputs.npz","golden.npy","goldens.npz","extra.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    with np.load(capture / "goldens.npz",allow_pickle=False) as expected:
        for path in (tmp_path / "qualification/native/outputs.npz",out / "consumer/host/outputs.npz"):
            with np.load(path,allow_pickle=False) as actual:
                assert actual.files == ["out0","out1","out2"]
                for name in actual.files:
                    assert actual[name].dtype == expected[name].dtype and actual[name].shape == expected[name].shape
                    assert actual[name].tobytes() == expected[name].tobytes()
    host = json.loads((out / "consumer/host/execution.json").read_text())
    assert host["whole_body_lowering"] == host["compiler_input_staging"] == host["reference_comparison"] == "passed"
    from t2m.bundle import _record
    trace_bytes,receipt_bytes = (out / "kernel-bindings.json").read_bytes(),(out / "capture_receipt.json").read_bytes()
    try:
        for change in ("schema","profile","policy","downgrade"):
            broken = deepcopy(trace)
            if change == "schema": broken["schema"] = "unknown"
            elif change == "profile": broken.pop("tensor_view_binding_profile")
            elif change == "policy": broken["calls"][0]["tensor_views"] = "unknown"
            else: broken["schema"] = "t2m.model-kernel-binding.v1"
            (out / "kernel-bindings.json").write_text(json.dumps(broken))
            receipt = json.loads(receipt_bytes)
            receipt["artifacts"]["kernel-bindings.json"] = _record(out / "kernel-bindings.json")
            (out / "capture_receipt.json").write_text(json.dumps(receipt))
            with pytest.raises(ValueError,match="binding|profile"):
                verify_bundle(out)
    finally:
        (out / "kernel-bindings.json").write_bytes(trace_bytes)
        (out / "capture_receipt.json").write_bytes(receipt_bytes)
    from t2m.allocation_bindings import ARGUMENTS_SCHEMA
    from t2m.cli import main
    arguments = {"schema":ARGUMENTS_SCHEMA,"source_entry":"aliased_updates","arguments":[
        *({"kind":"pointer","allocation":"state","offset":name} for name in ("read","first","second")),
        {"kind":"scalar","name":"n"},{"kind":"scalar","name":"steps"}]}
    configuration = {"schema":"t2m.kernel-bindings.v1","kernels":[{
        "kernel":binding.qualname,"ttir":"source.ttir","abi":result.spec.as_dict(),"allocation_bindings":arguments,
        "tensor_views":POLICY,"scalar_arguments":binding.scalar_arguments,"runtime_arguments":binding.runtime_arguments,
        "memory_profile":"guarded-buffer","address_policy":"source-words"} for binding in bindings]}
    (tmp_path / "bindings.json").write_text(json.dumps(configuration))
    cli_out = tmp_path / "cli-bound"
    assert main(["bind-model",str(capture),"--bindings",str(tmp_path / "bindings.json"),"--out",str(cli_out),
        "--optimizer",OPT,"--python",NATIVE,"--merlin-root",os.environ["T2M_MERLIN_ROOT"],"--merlin-python",
        os.environ["T2M_MERLIN_PYTHON"],"--merlin-clang",os.environ["T2M_MERLIN_CLANG"],"--merlin-host"]) == 0
    verify_bundle(cli_out)
    cli_trace = json.loads((cli_out / "kernel-bindings.json").read_text())
    assert len(cli_trace["kernels"]) == 1 and len({call["source_sha256"] for call in cli_trace["calls"]}) == 1


@pytest.mark.parametrize("dtype,kind",[("f32","dot"),("f16","half-dot"),("bf16","bfloat-dot")])
def test_real_tiled_dot_model_preserves_shared_C_weights_empty_K_and_all_results(tmp_path,dtype,kind):
    from t2m.allocation_state_demo import convert
    from t2m.bundle import verify_bundle
    from t2m.model_binding import KernelBinding
    from t2m.model_bundle import write_model_bundle
    from t2m.model_tensor_views import POLICY
    capture = tmp_path / "capture"
    script = Path(__file__).resolve().parents[1] / "scripts/capture_allocation_state_model.py"
    process = subprocess.run([M2M,str(script),"--out",str(capture),"--kind",kind],capture_output=True,text=True,timeout=60,check=False,
        env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TMPDIR=str(tmp_path)))
    assert process.returncode == 0,process.stderr
    result = convert(tmp_path / "source",dtype=dtype,optimizer=OPT)
    bindings = []
    for k in (37,0):
        m,n = 17,19
        values = (m,n,k,k,1,n,1,0,0,m*k,k*n,n,1,0,m*n,0)
        scalars = dict(zip((scalar.name for scalar in result.spec.runtime_scalars),values,strict=True))
        bindings.append(KernelBinding("t2m_demo::allocation_dot",result,tensor_views=POLICY,
            operands={"x":"x","y":"weight","state":"state"},scalar_arguments=scalars,
            runtime_arguments={scalar.name:scalar.name for scalar in result.spec.runtime_scalars}))
    out = tmp_path / "bound"
    write_model_bundle(capture,bindings,out,work=tmp_path / "qualification",python=NATIVE,
        merlin_root=os.environ["T2M_MERLIN_ROOT"],merlin_python=os.environ["T2M_MERLIN_PYTHON"],
        merlin_host=True,merlin_clang=os.environ["T2M_MERLIN_CLANG"])
    verify_bundle(out)
    trace = json.loads((out / "kernel-bindings.json").read_text())
    assert trace["schema"] == "t2m.model-kernel-binding.v2" and len(trace["calls"]) == 2 and len(trace["kernels"]) == 1
    assert (out / "model.mlir").read_text().count('"linalg.matmul"') == 2
    for name in ("weights.safetensors","weights.safetensors.manifest.json","inputs.npz","golden.npy","goldens.npz","extra.npz"):
        assert (out / name).read_bytes() == (capture / name).read_bytes()
    with np.load(capture / "goldens.npz",allow_pickle=False) as expected:
        for path in (tmp_path / "qualification/native/outputs.npz",out / "consumer/host/outputs.npz"):
            with np.load(path,allow_pickle=False) as actual:
                for name in expected.files:
                    assert actual[name].dtype == expected[name].dtype and actual[name].shape == expected[name].shape
                    assert actual[name].tobytes() == expected[name].tobytes()
    if (capture / "tensor_words.npz").exists():
        assert (out / "tensor_words.npz").read_bytes() == (capture / "tensor_words.npz").read_bytes()
