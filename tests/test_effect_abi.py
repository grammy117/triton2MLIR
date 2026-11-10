"""Explicit void source signatures retain effects without artificial results."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared


def test_legacy_abis_do_not_implicitly_admit_zero_results_or_change_serialization():
    abi = TensorABI("copy",(TensorBuffer("x",(1,)),TensorBuffer("out",(1,),"readwrite")))
    assert "effect_only" not in abi.as_dict()
    with pytest.raises(ValueError):
        TensorABI("check",())
    with pytest.raises(ValueError):
        TensorABI("check",(TensorBuffer("x",(1,)),))
    with pytest.raises(ValueError,match="readonly"):
        TensorABI("check",(TensorBuffer("out",(1,),"write"),),effect_only=True)
    spec = TensorABI("check",(),effect_only=True).as_dict()
    assert TensorABI.from_dict(spec).outputs == ()
    spec["schema"] = "t2m.shared-tensor-abi.v1"
    with pytest.raises(ValueError,match="v6"):
        TensorABI.from_dict(spec)


OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"),os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = pytest.mark.skipif(not (OPT and NATIVE),reason="existing native toolchain required")


@REQUIRED
@pytest.mark.parametrize("profile", ["tensor","buffer","guarded-buffer"])
def test_readonly_dynamic_tensor_checks_return_no_values(tmp_path,profile):
    from t2m.examples.effect_kernels import check_tensor
    from t2m.frontend import capture_ttir
    from t2m.native import execute_native
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(check_tensor,{"x":"*i32","n":"i32","limit":"i32","B":"constexpr"},{"B":8},debug=True))
    abi = TensorABI("check_tensor",(TensorBuffer("x",("n",),dtype="i32"),),grid=(GridExtent("n",8),),
        runtime_scalars=(RuntimeScalar("n","i32",1,bounds=(0,17)),RuntimeScalar("limit","i32",2,bounds=(-9,9))),effect_only=True)
    assert TensorABI.from_dict(abi.as_dict()) == abi
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    assert len(next(iter(result.module.ops)).function_type.outputs) == 0
    for n in (0,1,13,17):
        outputs,_ = execute_native(result.mlir_text,[np.arange(n,dtype=np.int32)%5,np.asarray(n,np.int32),np.asarray(5,np.int32)],
            tmp_path / f"native-{n}",python=NATIVE,shape_abi=abi.runtime_shapes())
        assert outputs == ()
