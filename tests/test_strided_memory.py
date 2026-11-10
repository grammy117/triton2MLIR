"""Non-dense source allocations retain K loops and reuse through Merlin."""

import json
import os
import signal

import numpy as np
import pytest
from xdsl.dialects.builtin import TensorType, f32

from t2m.examples.strided_memory_reference import arguments,policy
from t2m.strided_memory_demo import convert

OPT,NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = OPT and NATIVE and all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED,reason="existing shared/native/Merlin environments required")


@pytest.mark.parametrize("dtype",["f32","f16","bf16"])
def test_signed_padded_transposed_and_broadcast_views_reuse_one_merlin_library(tmp_path,dtype):
    from test_mixed_effects import qualify
    from t2m.invocation import invoke_kernel,verify_invocation

    result = convert(tmp_path / "source",dtype=dtype,optimizer=OPT)
    matmuls = [op for op in result.module.walk() if op.name == "linalg.matmul"]
    assert len(matmuls) == 1
    ancestors = []
    parent = matmuls[0].parent_op()
    while parent is not None:
        ancestors.append(parent)
        parent = parent.parent_op()
    assert any(op.name == "scf.for" and any(value.type == TensorType(f32,(16,16)) for value in op.results) for op in ancestors)
    assert "scf.for" in result.mlir_text
    assert "tensor<?x" in result.mlir_text
    function = next(iter(result.module.ops))
    assert "source allocations" in function.attributes["prov.intent"].data
    assert result.trace["intent"]["grants_semantics"] is False
    values,expected = arguments(17,19,37,(79,2,64,3),dtype)
    _,qualified = qualify(result,tmp_path,values,policy(),[expected])
    cases = [
        (17,19,37,(1,20,1,42),False),  # column-major source views
        (17,19,37,(-79,-2,-64,-3),False),
        (17,19,37,(79,-2,-64,3),False),
        (17,19,37,(0,2,3,0),False),  # readonly repeated/broadcast addresses
        (35,21,65,(135,2,70,3),False),
        (1,1,65,(-70,1,3,-2),False),
        (1,1,32,(70,-1,-3,2),True),  # C must remain inside the contraction
        (5,7,0,(-256,256,256,-256),False),
        (0,7,65,(70,1,20,1),False),
        (7,0,65,(70,1,20,1),False),
    ]
    for index,(m,n,k,strides,sensitive) in enumerate(cases):
        values,expected = arguments(m,n,k,strides,dtype,sensitive=sensitive)
        if k == 0:
            # These origins would be out of bounds for an actual access, but
            # the original zero-trip K loop performs no input observation.
            values[10] = np.asarray(8191,np.int64)
            values[11] = np.asarray(8191,np.int64)
        if sensitive:
            np.testing.assert_array_equal(expected,np.full((m,n),2.**24,np.float32))
        caller = tmp_path / f"caller-{index}";caller.mkdir()
        np.savez(caller / "inputs.npz",**{f"in{i}":value for i,value in enumerate(values)})
        np.savez(caller / "reference.npz",out0=expected)
        (caller / "reference.json").write_text(json.dumps(policy()))
        out = tmp_path / f"invoked-{index}"
        report = invoke_kernel(qualified,caller / "inputs.npz",caller / "reference.json",out,
                               merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
    for index,argument in enumerate((10,11)):
        values,expected = arguments(3,5,7,(19,2,23,3),dtype)
        values[argument] = np.asarray(8191,np.int64)
        caller = tmp_path / f"bad-{index}";caller.mkdir()
        np.savez(caller / "inputs.npz",**{f"in{i}":value for i,value in enumerate(values)})
        np.savez(caller / "reference.npz",out0=expected)
        (caller / "reference.json").write_text(json.dumps(policy()))
        out = tmp_path / f"failed-{index}"
        # Compiler guard failures must remain refused invocations. They cannot
        # acquire the successful source-assertion outcome contract.
        with pytest.raises(ValueError,match="actual effect outcome differs"):
            invoke_kernel(qualified,caller / "inputs.npz",caller / "reference.json",out,
                          merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"])
        assert json.loads((out / "evaluation/outcome.json").read_text())["returncode"] == -signal.SIGABRT
        assert (out / "evaluation/invocation-started.json").is_file()
        assert (out / "evaluation/stdout.bin").read_bytes() == b"Triton active memory index outside logical allocation\n"
        assert (out / "source/host/model.so").read_bytes() == (qualified / "host/model.so").read_bytes()
        assert not (out / "invocation.json").exists()
        assert not (out / "evaluation/outputs.npz").exists()
