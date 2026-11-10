"""Readonly aliases reach both package writers and the real Merlin interface."""

import json
import os

import numpy as np
import pytest

from t2m.effect_reference import validate_inputs
from t2m.input_storage import READONLY_SNAPSHOTS

OPT,NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = OPT and NATIVE and all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED,reason="existing shared/native/Merlin environments required")


@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
def test_numeric_writer_snapshots_overlapping_readonly_arguments(tmp_path,profile):
    from test_shared_tensor import source_case

    from t2m.bundle import verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    from t2m.shared_tensor import convert_shared

    source,abi = source_case("matrix_add",tmp_path)
    result = convert_shared(source,abi,tmp_path / "convert",optimizer=OPT,memory_profile=profile)
    owner = np.arange(64,dtype=np.float32)
    x,y = owner[:32].reshape(4,8),owner[16:48].reshape(4,8)
    assert np.shares_memory(x,y)
    before = owner.tobytes()
    old = np.full((4,8),-19.,np.float32)
    capture,qualified = tmp_path / "capture",tmp_path / "qualified"
    write_shared_bundle(result,capture,[x,y,old],[x+y],work=tmp_path / "native",python=NATIVE)
    verify_bundle(capture)
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
                   compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    verify_kernel_qualification(qualified)
    assert owner.tobytes() == before
    with pytest.raises(ValueError,match="writable source storage"):
        write_shared_bundle(result,tmp_path / "bad-capture",[x,y,x],[x+y],work=tmp_path / "bad-native",python=NATIVE)
    assert not (tmp_path / "bad-native").exists() and not (tmp_path / "bad-capture").exists()
    assert result.trace["memory"]["input_preparation"] == READONLY_SNAPSHOTS
    with np.load(capture / "inputs.npz",allow_pickle=False) as inputs:
        assert inputs["in0"].tobytes() == x.tobytes() and inputs["in1"].tobytes() == y.tobytes()
    for index,offset in enumerate((0,16,32)):
        values = [owner[:32].reshape(4,8),owner[offset:offset+32].reshape(4,8),old]
        prepared = validate_inputs(abi,values)
        assert not np.shares_memory(prepared[0],prepared[1])
        caller = tmp_path / f"caller-{index}";caller.mkdir()
        np.savez(caller / "inputs.npz",**{f"in{i}":value for i,value in enumerate(prepared)})
        np.savez(caller / "reference.npz",out0=values[0]+values[1])
        out = tmp_path / f"invoke-{index}"
        report = invoke_kernel(qualified,caller / "inputs.npz",caller / "reference.npz",out,
                               merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
    assert owner.tobytes() == before


@pytest.mark.parametrize("dtype",["f32","f16","bf16"])
def test_aliased_operand_spans_and_independent_writable_state_reuse_merlin_library(tmp_path,dtype):
    from test_mixed_effects import qualify

    from t2m.effect_bundle import write_effect_bundle
    from t2m.examples.readonly_alias_reference import aliased_arguments
    from t2m.examples.strided_memory_reference import policy
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.strided_memory_demo import convert

    result = convert(tmp_path / "source",dtype=dtype,optimizer=OPT,writable=True)
    values,expected,owner = aliased_arguments(17,19,37,(79,2,64,3),dtype,writable=True)
    assert np.shares_memory(values[0],values[1])
    before = owner.tobytes()
    _,qualified = qualify(result,tmp_path,values,policy(words=True),[expected])
    assert owner.tobytes() == before
    bad = [value.copy() for value in values]
    storage = np.zeros(max(bad[0].nbytes,bad[3].nbytes),np.uint8)
    bad[0] = storage[:bad[0].nbytes].view(bad[0].dtype).reshape(bad[0].shape)
    bad[3] = storage[:bad[3].nbytes].view(bad[3].dtype).reshape(bad[3].shape)
    with pytest.raises(ValueError,match="writable source storage"):
        write_effect_bundle(result,tmp_path / "bad-capture",bad,policy(words=True),references=[expected],
                            work=tmp_path / "bad-native",python=NATIVE)
    assert not (tmp_path / "bad-native").exists() and not (tmp_path / "bad-capture").exists()
    for index,(m,n,k,strides,offset) in enumerate([
        (17,19,37,(1,20,1,42),5),
        (17,19,37,(-79,-2,-64,-3),11),
        (3,5,65,(0,2,3,0),5),
        (5,7,0,(-256,256,256,-256),11),
        (0,7,65,(70,1,20,1),11),
    ]):
        values,expected,owner = aliased_arguments(m,n,k,strides,dtype,writable=True,offset=offset)
        before = owner.tobytes()
        prepared = validate_inputs(result.spec,values)
        assert not np.shares_memory(prepared[0],prepared[1])
        for original,snapshot in zip(values,prepared,strict=True):
            assert snapshot.tobytes() == original.tobytes()
        caller = tmp_path / f"caller-{index}";caller.mkdir()
        np.savez(caller / "inputs.npz",**{f"in{i}":value for i,value in enumerate(prepared)})
        np.savez(caller / "reference.npz",out0=expected)
        (caller / "reference.json").write_text(json.dumps(policy(words=True)))
        out = tmp_path / f"invoke-{index}"
        report = invoke_kernel(qualified,caller / "inputs.npz",caller / "reference.json",out,
                               merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library" and owner.tobytes() == before


@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
def test_by_value_scalar_alias_of_writable_state_preserves_its_initial_value(tmp_path,profile):
    from test_mixed_effects import qualify

    from t2m.examples.readonly_alias_kernels import scalar_carrier_update
    from t2m.frontend import capture_ttir
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(scalar_carrier_update,{"state":"*i64","step":"i32","N":"constexpr"},{"N":8}))
    abi = TensorABI("scalar_carrier_update",(TensorBuffer("state",(8,),"readwrite","i64"),),
                    runtime_scalars=(RuntimeScalar("step","i32",1,bounds=(-9,9)),))
    result = convert_shared(source,abi,tmp_path / "convert",optimizer=OPT,memory_profile=profile,
                            address_policy="source-words" if profile == "guarded-buffer" else "structured")
    policy = {"schema":"t2m.effect-reference.v2","termination":"return","stdout":"","stderr":"",
              "results":{"archive":"reference.npz","comparisons":[{"kind":"words"}]}}
    state = np.arange(8,dtype=np.int64)+2**53+1
    state[0] = 3
    scalar = state.view(np.int32)[:1].reshape(())
    before = state.tobytes()
    assert np.shares_memory(state,scalar) and scalar.item() == 3
    _,qualified = qualify(result,tmp_path,[state,scalar],policy,[state+3])
    assert state.tobytes() == before
    for index,value in enumerate((-2,0,7)):
        state[0] = value
        before = state.tobytes()
        expected = state+value
        prepared = validate_inputs(abi,[state,scalar])
        assert prepared[0] is state and prepared[1].item() == value
        caller = tmp_path / f"caller-{index}";caller.mkdir()
        np.savez(caller / "inputs.npz",in0=prepared[0],in1=prepared[1])
        np.savez(caller / "reference.npz",out0=expected)
        (caller / "reference.json").write_text(json.dumps(policy))
        out = tmp_path / f"invoke-{index}"
        report = invoke_kernel(qualified,caller / "inputs.npz",caller / "reference.json",out,
                               merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library" and state.tobytes() == before
