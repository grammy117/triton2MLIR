"""Strided output state preserves every untouched word and rejects races."""

import json
import os
import signal

import numpy as np
import pytest
from xdsl.dialects.builtin import TensorType, f32

from t2m.examples.strided_memory_reference import output_arguments,policy
from t2m.strided_memory_demo import convert

OPT,NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = OPT and NATIVE and all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED,reason="existing shared/native/Merlin environments required")


def caller(root,values,expected):
    root.mkdir()
    np.savez(root / "inputs.npz",**{f"in{i}":value for i,value in enumerate(values)})
    np.savez(root / "reference.npz",out0=expected)
    (root / "reference.json").write_text(json.dumps(policy(words=True)))
    return root / "inputs.npz",root / "reference.json"


@pytest.mark.parametrize("dtype",["f32","f16","bf16"])
def test_strided_partial_writes_reuse_library_preserve_holes_and_reject_conflicts(tmp_path,dtype):
    from test_mixed_effects import qualify
    from t2m.invocation import invoke_kernel,verify_invocation

    result = convert(tmp_path / "source",dtype=dtype,optimizer=OPT,writable=True)
    matmuls = [op for op in result.module.walk() if op.name == "linalg.matmul"]
    assert len(matmuls) == 1
    parent = matmuls[0].parent_op()
    assert parent.name == "scf.for" and any(value.type == TensorType(f32,(16,16)) for value in parent.results)
    assert result.spec.outputs[0].effect == "readwrite"
    assert "preserve-incoming-output-state" in result.mlir_text
    values,expected = output_arguments(17,19,37,(79,2,64,3),(-47,2),dtype)
    _,qualified = qualify(result,tmp_path,values,policy(words=True),[expected])
    cases = [
        (17,19,37,(1,20,1,42),(1,21),False),
        (17,19,37,(-79,-2,-64,-3),(-47,-2),False),
        (17,19,37,(79,-2,-64,3),(47,-2),False),
        (17,19,37,(0,2,3,0),(45,2),False),
        (35,21,65,(135,2,70,3),(70,3),False),
        (1,1,32,(70,-1,-3,2),(-7,-3),True),
        (5,7,0,(-256,256,256,-256),(-23,3),False),
        (0,7,65,(70,1,20,1),(-23,3),False),
        (7,0,65,(70,1,20,1),(-23,3),False),
    ]
    for index,(m,n,k,strides,out_strides,sensitive) in enumerate(cases):
        values,expected = output_arguments(m,n,k,strides,out_strides,dtype,sensitive=sensitive)
        if not m or not n:
            values[17] = np.asarray(8191,np.int64)  # Unobserved output origin.
            assert expected.tobytes() == values[3].tobytes()
        paths = caller(tmp_path / f"caller-{index}",values,expected)
        out = tmp_path / f"invoked-{index}"
        report = invoke_kernel(qualified,*paths,out,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
        with np.load(out / "evaluation/outputs.npz",allow_pickle=False) as actual:
            assert actual["out0"].tobytes() == expected.tobytes()
    failures = [
        (3,5,(0,1),"Triton scatter has conflicting active lanes"),
        (17,2,(1,16),"Triton memory conflicts across logical programs"),
        (3,5,(23,3),"Triton active memory index outside logical allocation"),
    ]
    for index,(m,n,strides,message) in enumerate(failures):
        values,expected = output_arguments(m,n,7,(19,2,23,3),strides,dtype,conflicts=index < 2)
        if index == 2:
            values[17] = np.asarray(8191,np.int64)
        paths = caller(tmp_path / f"bad-{index}",values,expected)
        out = tmp_path / f"failed-{index}"
        with pytest.raises(ValueError,match="actual effect outcome differs"):
            invoke_kernel(qualified,*paths,out,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        assert json.loads((out / "evaluation/outcome.json").read_text())["returncode"] == -signal.SIGABRT
        assert (out / "evaluation/invocation-started.json").is_file()
        assert (out / "evaluation/stdout.bin").read_bytes() == (message+"\n").encode()
        assert (out / "source/host/model.so").read_bytes() == (qualified / "host/model.so").read_bytes()
        assert not (out / "invocation.json").exists() and not (out / "evaluation/outputs.npz").exists()
