"""Void and initially failing packages cross the actual Merlin compiler ABI."""

import json
import os

import numpy as np

import pytest

from test_effect_bundle import convert,reference,NATIVE,OPT
from t2m.effect_bundle import write_effect_bundle
from t2m.kernel_consumer import qualify_kernel,verify_kernel_qualification

REQUIRED = all(os.environ.get(k) for k in ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG")) and OPT and NATIVE
pytestmark = pytest.mark.skipif(not REQUIRED,reason="existing native/Merlin environments required")


@pytest.mark.parametrize("profile", ["tensor","buffer","guarded-buffer"])
@pytest.mark.parametrize("fail", [False,True])
def test_real_zero_argument_merlin_entry_matches_return_or_source_assertion(tmp_path,profile,fail):
    result = convert(tmp_path,profile,fail=fail)
    capture,qualified = tmp_path / "capture",tmp_path / "merlin"
    write_effect_bundle(result,capture,[],reference(fail),work=tmp_path / "native",python=NATIVE)
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    receipt = verify_kernel_qualification(qualified)
    assert receipt["schema"] == "t2m.merlin-effect-qualification.v1"
    signature = json.loads((qualified / "compiler/signature.json").read_text())
    assert signature["inputs"] == signature["outputs"] == []
    report = json.loads((qualified / "host/execution.json").read_text())
    assert report["reference_comparison"] == "passed" and report["termination"] == reference(fail)["termination"]
    assert report["declared_result_count"] == report["result_count"] == 0
    assert (qualified / "evaluation/outputs.npz").exists() == (not fail)


def test_initial_failure_keeps_the_real_tensor_result_signature_at_merlin(tmp_path):
    from t2m.examples.effect_kernels import fail_before_store
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorABI,TensorBuffer,convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(fail_before_store,{"out":"*i32","B":"constexpr"},{"B":8},debug=True))
    abi = TensorABI("fail_before_store",(TensorBuffer("out",(8,),"write","i32"),))
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT)
    capture,qualified = tmp_path / "capture",tmp_path / "merlin"
    write_effect_bundle(result,capture,[],reference(True,"failure precedes real output"),work=tmp_path / "native",python=NATIVE)
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    signature = json.loads((qualified / "compiler/signature.json").read_text())
    assert signature["outputs"] == [{"shape":[8],"dtype":"i32"}]
    report = json.loads((qualified / "host/execution.json").read_text())
    assert report["declared_result_count"] == 1 and report["result_count"] == 0 and report["outputs_published"] is False
    # Rehashing a dropped declaration must not erase the real public result.
    from t2m.bundle import _record
    signature["outputs"] = []
    (qualified / "compiler/signature.json").write_text(json.dumps(signature))
    stage = json.loads((qualified / "compiler/compile-inputs.json").read_text())
    stage["members"]["signature.json"] = _record(qualified / "compiler/signature.json")["sha256"]
    (qualified / "compiler/compile-inputs.json").write_text(json.dumps(stage))
    receipt = json.loads((qualified / "qualification.json").read_text())
    for name in ("compiler/signature.json","compiler/compile-inputs.json"):
        receipt["artifacts"][name] = _record(qualified / name)
    (qualified / "qualification.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="signature differs"):
        verify_kernel_qualification(qualified)


def test_one_real_void_library_handles_new_scalar_values_and_expected_failures(tmp_path):
    from t2m.invocation import invoke_kernel,verify_invocation
    result = convert(tmp_path,"tensor",scalar=True)
    capture,qualified = tmp_path / "capture",tmp_path / "merlin"
    write_effect_bundle(result,capture,[np.asarray(-1,np.int32),np.asarray(3,np.int32)],
        reference(True,"effect scalar must be nonnegative"),work=tmp_path / "native",python=NATIVE)
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    original = (qualified / "qualification.json").read_bytes()
    for i,(value,steps,fail) in enumerate([(-1,0,False),(3,3,False),(-1,3,True)]):
        args,ref = tmp_path / f"arguments-{i}.npz",tmp_path / f"reference-{i}.json"
        np.savez(args,in0=np.asarray(value,np.int32),in1=np.asarray(steps,np.int32))
        ref.write_text(json.dumps(reference(fail,"effect scalar must be nonnegative")))
        out = tmp_path / f"invoke-{i}"
        report = invoke_kernel(qualified,args,ref,out,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["reference_comparison"] == "passed" and report["native_compilation"] == "reused-qualified-library"
        verify_invocation(out)
    assert (qualified / "qualification.json").read_bytes() == original


@pytest.mark.parametrize("profile", ["tensor","buffer","guarded-buffer"])
def test_one_void_tensor_library_handles_empty_tail_full_domains_and_failures(tmp_path,profile):
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.effect_kernels import check_tensor
    from t2m.frontend import capture_ttir
    from t2m.invocation import invoke_kernel,verify_invocation
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI,TensorBuffer,convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(check_tensor,{"x":"*i32","n":"i32","limit":"i32","B":"constexpr"},{"B":8},debug=True))
    abi = TensorABI("check_tensor",(TensorBuffer("x",("n",),dtype="i32"),),grid=(GridExtent("n",8),),
        runtime_scalars=(RuntimeScalar("n","i32",1,bounds=(0,17)),RuntimeScalar("limit","i32",2,bounds=(-9,9))),effect_only=True)
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    capture,qualified = tmp_path / "capture",tmp_path / "merlin"
    write_effect_bundle(result,capture,[np.arange(13,dtype=np.int32)%5,np.asarray(13,np.int32),np.asarray(5,np.int32)],
        reference(),work=tmp_path / "native",python=NATIVE)
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    for i,(n,limit) in enumerate([(0,3),(1,3),(13,3),(17,5)]):
        values = np.arange(n,dtype=np.int32)%5
        args,ref = tmp_path / f"arguments-{i}.npz",tmp_path / f"reference-{i}.json"
        np.savez(args,in0=values,in1=np.asarray(n,np.int32),in2=np.asarray(limit,np.int32))
        ref.write_text(json.dumps(reference(bool(np.any(values > limit)),"effect tile exceeds limit")))
        out = tmp_path / f"invoke-{i}"
        invoke_kernel(qualified,args,ref,out,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
