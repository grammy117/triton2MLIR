"""Explicit pointer sharing must retain source read-after-write dependencies."""

import hashlib
import json
import os
import signal

import numpy as np
import pytest

from t2m.allocation_bindings import PointerArgument, ScalarArgument, convert_allocations
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer

OPT,NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = OPT and NATIVE and all(os.environ.get(name) for name in
    ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED,reason="existing shared/native/Merlin environments required")


def convert(root, *, profile="guarded-buffer", dynamic=True):
    from t2m.examples.readonly_alias_kernels import aliased_updates
    source = root / "source.ttir"
    source.write_text(capture_ttir(aliased_updates,{"read":"*i64","first":"*i64","second":"*i64",
        "n":"i32","steps":"i32","B":"constexpr"},{"B":8}))
    scalars = (RuntimeScalar("n","i32",1,bounds=(0,17)),RuntimeScalar("steps","i32",2,bounds=(0,3)))
    if dynamic:
        scalars += tuple(RuntimeScalar(name,"i64",index,bounds=(-32,63)) for index,name in enumerate(("read","first","second"),3))
    abi = TensorABI("allocation_updates",(TensorBuffer("state",(48,),"readwrite","i64"),),
        runtime_scalars=scalars,grid=(GridExtent("n",8),) if dynamic else (1,),
        intent="Source alias dependencies and iteration carried updates; one owning root state")
    pointers = tuple(PointerArgument("state",name if dynamic else 0) for name in ("read","first","second"))
    return convert_allocations(source,abi,(*pointers,ScalarArgument("n"),ScalarArgument("steps")),
        root / "conversion",source_entry="aliased_updates",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")


def arguments(n,steps,origins=(0,0,0),*,dynamic=True):
    state = np.arange(48,dtype=np.int64)+2**53+1
    expected = state.copy()
    read,first,second = origins
    # A source load observes a whole tile before its following store. All
    # partial-overlap cases below have one logical program, avoiding GPU races.
    for iteration in range(steps):
        value = expected[read:read+n].copy()
        expected[first:first+n] = value+3
        forwarded = expected[read:read+n].copy()
        expected[second:second+n] = forwarded*2
    values = [state,np.asarray(n,np.int32),np.asarray(steps,np.int32)]
    if dynamic:
        values += [np.asarray(value,np.int64) for value in origins]
    return values,expected


def policy():
    return {"schema":"t2m.effect-reference.v2","termination":"return","stdout":"","stderr":"",
        "results":{"archive":"reference.npz","comparisons":[{"kind":"words"}]}}


def replay(qualified,root,values,expected):
    from t2m.invocation import invoke_kernel, verify_invocation
    root.mkdir()
    np.savez(root / "inputs.npz",**{f"in{i}":value for i,value in enumerate(values)})
    np.savez(root / "reference.npz",out0=expected)
    (root / "reference.json").write_text(json.dumps(policy()))
    before = [value.tobytes() for value in values]
    report = invoke_kernel(qualified,root / "inputs.npz",root / "reference.json",root / "invocation",
        merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
    verify_invocation(root / "invocation")
    assert report["native_compilation"] == "reused-qualified-library"
    assert [value.tobytes() for value in values] == before


def test_runtime_partial_aliases_and_shared_updates_reuse_real_merlin_library(tmp_path):
    from test_mixed_effects import qualify
    result = convert(tmp_path)
    values,expected = arguments(7,2,(0,4,8))
    capture,qualified = qualify(result,tmp_path,values,policy(),[expected])
    assert len(result.spec.outputs) == 1 and len(result.spec.inputs) == 6
    assert "tt.call" in {row["opcode"] for row in result.trace["source_operations"]}
    assert "scf.for" in result.mlir_text and "source-allocation-binding" in result.source_ttir
    assert any("public allocation-state entry" in text for text in
        json.loads((capture / "compatibility.json").read_text())["differences"])
    before = (qualified / "host/model.so").read_bytes()
    for index,(n,steps,origins) in enumerate([
        (17,2,(0,0,0)),(7,2,(4,0,8)),(5,3,(8,4,0)),(3,1,(20,21,22)),
        (0,3,(63,-32,63)),(17,0,(63,-32,63)),
    ]):
        values,expected = arguments(n,steps,origins) if n and steps else arguments(n,0,origins)
        if not steps or not n:
            values[2] = np.asarray(steps,np.int32)
        replay(qualified,tmp_path / f"case-{index}",values,expected)
    assert (qualified / "host/model.so").read_bytes() == before
    from t2m.invocation import invoke_kernel
    for index,(n,origins,message) in enumerate([
        (3,(47,0,0),"Triton active memory index outside logical allocation"),
        (17,(0,4,8),"Triton memory conflicts across logical programs"),
    ]):
        values,expected = arguments(n,0,origins)
        values[2] = np.asarray(1,np.int32)
        caller = tmp_path / f"invalid-{index}";caller.mkdir()
        np.savez(caller / "inputs.npz",**{f"in{i}":value for i,value in enumerate(values)})
        np.savez(caller / "reference.npz",out0=expected)
        (caller / "reference.json").write_text(json.dumps(policy()))
        out = caller / "invocation"
        with pytest.raises(ValueError,match="actual effect outcome differs"):
            invoke_kernel(qualified,caller / "inputs.npz",caller / "reference.json",out,
                merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        assert json.loads((out / "evaluation/outcome.json").read_text())["returncode"] == -signal.SIGABRT
        assert (out / "evaluation/stdout.bin").read_bytes() == (message+"\n").encode()
        assert (out / "evaluation/invocation-started.json").is_file()
        assert not (out / "evaluation/outputs.npz").exists() and not (out / "invocation.json").exists()
        assert (out / "source/host/model.so").read_bytes() == before


@pytest.mark.parametrize("profile",["tensor","buffer","guarded-buffer"])
def test_identical_source_pointers_use_one_state_in_every_memory_profile(tmp_path,profile):
    from test_mixed_effects import qualify
    result = convert(tmp_path,profile=profile,dynamic=False)
    values,expected = arguments(7,2,dynamic=False)
    _,qualified = qualify(result,tmp_path,values,policy(),[expected])
    replay(qualified,tmp_path / "replay",*arguments(5,3,dynamic=False))


def test_saved_bindings_compile_a_native_checked_package_through_cli(tmp_path):
    from t2m.allocation_bindings import ARGUMENTS_SCHEMA
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    from t2m.examples.readonly_alias_kernels import aliased_updates
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(aliased_updates,{"read":"*i64","first":"*i64","second":"*i64",
        "n":"i32","steps":"i32","B":"constexpr"},{"B":8}).replace("\n","\r\n"))
    abi = TensorABI("fixed_state",(TensorBuffer("state",(48,),"readwrite","i64"),))
    (tmp_path / "abi.json").write_text(json.dumps(abi.as_dict()))
    (tmp_path / "bindings.json").write_text(json.dumps({"schema":ARGUMENTS_SCHEMA,"source_entry":"aliased_updates",
        "arguments":[*({"kind":"pointer","allocation":"state"} for _ in range(3)),
                     {"kind":"constant","value":7},{"kind":"constant","value":2}]}))
    values,expected = arguments(7,2,dynamic=False)
    np.savez(tmp_path / "inputs.npz",in0=values[0])
    np.savez(tmp_path / "reference.npz",out0=expected)
    out = tmp_path / "cli-capture"
    assert main(["shared-tensor",str(source),"--abi",str(tmp_path / "abi.json"),"--allocation-bindings",
        str(tmp_path / "bindings.json"),"--memory-profile","guarded-buffer","--address-policy","source-words",
        "--optimizer",OPT,"--python",NATIVE,"--inputs",str(tmp_path / "inputs.npz"),"--reference",
        str(tmp_path / "reference.npz"),"--out",str(out)]) == 0
    verify_bundle(out)
    original = source.read_bytes()
    work = tmp_path / "cli-capture.work/translation"
    assert (work / "original.ttir").read_bytes() == original and b"\r\n" in original
    binding = json.loads((out / "provenance.json").read_text())["source_argument_binding"]
    assert binding["original_source_sha256"] == hashlib.sha256(original).hexdigest()
