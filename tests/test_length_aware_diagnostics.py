"""Original diagnostic bytes survive real TTIR, native and Merlin execution."""

import json
import os

import numpy as np
import pytest

from t2m.bundle import verify_bundle
from t2m.effect_bundle import write_effect_bundle
from t2m.frontend import capture_ttir
from t2m.invocation import invoke_kernel, verify_invocation
from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = OPT and NATIVE and all(os.environ.get(key) for key in
    ("T2M_MERLIN_ROOT","T2M_MERLIN_PYTHON","T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing native/Merlin environments required")

MESSAGE = 'before\x00after %s "value"\nλ\x00end'


def reference(fail=False, *, message=MESSAGE, original_cf=False):
    return {"schema":"t2m.effect-reference.v1", "termination":"assertion_failure" if fail else "return",
        "stdout":((message if original_cf else "Assertion `"+message+"` failed")+"\n") if fail else "",
        "stderr":""}


@pytest.mark.parametrize("profile", ["tensor","buffer","guarded-buffer"])
@pytest.mark.parametrize("kind", ["scalar","tensor"])
def test_source_bytes_reach_real_merlin_and_one_reused_library(tmp_path, profile, kind):
    from t2m.examples.effect_kernels import check_diagnostic, check_tensor_diagnostic
    from t2m.dynamic_abi import GridExtent
    source = tmp_path / "source.ttir"
    if kind == "scalar":
        kernel = check_diagnostic
        signature, constants = {"value":"i32","steps":"i32","MESSAGE":"constexpr"}, {"MESSAGE":MESSAGE}
        abi = TensorABI(kernel.__name__,(),runtime_scalars=(RuntimeScalar("value","i32",0,bounds=(-9,9)),
            RuntimeScalar("steps","i32",1,bounds=(0,4))),effect_only=True)
        initial = [np.asarray(-1,np.int32),np.asarray(3,np.int32)]
        variants = [([np.asarray(-1,np.int32),np.asarray(0,np.int32)],False),
                    ([np.asarray(3,np.int32),np.asarray(3,np.int32)],False)]
    else:
        kernel = check_tensor_diagnostic
        signature = {"x":"*i32","n":"i32","limit":"i32","MESSAGE":"constexpr","B":"constexpr"}
        constants = {"MESSAGE":MESSAGE,"B":8}
        abi = TensorABI(kernel.__name__,(TensorBuffer("x",("n",),dtype="i32"),),grid=(GridExtent("n",8),),
            runtime_scalars=(RuntimeScalar("n","i32",1,bounds=(0,17)),RuntimeScalar("limit","i32",2,bounds=(-9,9))),
            effect_only=True)
        initial = [np.arange(13,dtype=np.int32)%5,np.asarray(13,np.int32),np.asarray(3,np.int32)]
        variants = [([np.zeros(0,np.int32),np.asarray(0,np.int32),np.asarray(-1,np.int32)],False),
                    ([initial[0],initial[1],np.asarray(5,np.int32)],False)]
    source.write_text(capture_ttir(kernel,signature,constants,debug=True))
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT,memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    capture, qualified = tmp_path / "capture", tmp_path / "qualified"
    write_effect_bundle(result,capture,initial,reference(True),work=tmp_path / "native",python=NATIVE)
    verify_bundle(capture)
    assert (capture / "native/stdout.bin").read_bytes() == reference(True)["stdout"].encode("utf-8")
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    verify_kernel_qualification(qualified)
    assert (qualified / "evaluation/stdout.bin").read_bytes() == (capture / "native/stdout.bin").read_bytes()
    compilation = json.loads((qualified / "host/compilation.json").read_text())
    assert "lower_length_aware_diagnostics" in compilation["features"]
    assert "@putchar" in (qualified / "host/model.ll").read_text()
    assert "__merlin_assert_bytes_" in (qualified / "host/model.ll").read_text()
    variants.append((initial,True))
    for index,(arrays,fail) in enumerate(variants):
        arguments, expected = tmp_path / f"arguments-{index}.npz", tmp_path / f"reference-{index}.json"
        np.savez(arguments,**{f"in{i}":array for i,array in enumerate(arrays)})
        expected.write_text(json.dumps(reference(fail)))
        invocation = tmp_path / f"invoke-{index}"
        invoke_kernel(qualified,arguments,expected,invocation,merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(invocation)
        assert (invocation / "evaluation/stdout.bin").read_bytes() == reference(fail)["stdout"].encode("utf-8")


def test_original_cf_assert_and_wrong_truncated_reference(tmp_path):
    source = tmp_path / "source.ttir"
    source.write_text('''module {tt.func public @original(%condition:i1) {
      cf.assert %condition,"plain\\00message λ"
      tt.return
    }}''')
    abi = TensorABI("original",(),runtime_scalars=(RuntimeScalar("condition","i1",0),),effect_only=True)
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT)
    expected = reference(True,message="plain\x00message λ",original_cf=True)
    capture = tmp_path / "capture"
    write_effect_bundle(result,capture,[np.asarray(False)],expected,work=tmp_path / "native",python=NATIVE)
    qualified = tmp_path / "qualified"
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    verify_kernel_qualification(qualified)
    arguments, ref = tmp_path / "arguments.npz",tmp_path / "reference.json"
    np.savez(arguments,in0=np.asarray(False))
    ref.write_text(json.dumps(dict(expected,stdout="plain\n")))
    with pytest.raises(ValueError,match="actual source assertion"):
        invoke_kernel(qualified,arguments,ref,tmp_path / "truncated",merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"])


@pytest.mark.parametrize("kind", ["bfloat","atomic"])
def test_diagnostic_native_seam_composes_with_existing_transport_and_ownership(tmp_path,kind):
    from t2m.examples.effect_kernels import copy_diagnostic,atomic_diagnostic
    source = tmp_path / "source.ttir"
    options = {}
    if kind == "bfloat":
        kernel = copy_diagnostic
        signature = {"x":"*bf16","out":"*bf16","allow":"i1","MESSAGE":"constexpr","B":"constexpr"}
        constants = {"MESSAGE":MESSAGE,"B":8}
        abi = TensorABI(kernel.__name__,(TensorBuffer("x",(8,),dtype="bf16"),TensorBuffer("out",(8,),"write","bf16")),
            runtime_scalars=(RuntimeScalar("allow","i1",2),))
        arrays = [np.asarray([0x7f81,0x8000,1,0x3f80,0xff80,0x7f80,0x4000,0x0000],np.uint16),np.asarray(False)]
        feature = "preserve_bfloat_transport"
    else:
        kernel = atomic_diagnostic
        signature = {"out":"*i32","allow":"i1","MESSAGE":"constexpr"}
        constants = {"MESSAGE":MESSAGE}
        abi = TensorABI(kernel.__name__,(TensorBuffer("out",(1,),"readwrite","i32"),),
            runtime_scalars=(RuntimeScalar("allow","i1",1),))
        arrays = [np.asarray([1],np.int32),np.asarray(False)]
        options = {"memory_profile":"guarded-buffer","address_policy":"source-words","effect_profile":"typed-atomics"}
        feature = "lower_integer_compare_exchange"
    source.write_text(capture_ttir(kernel,signature,constants,debug=True))
    result = convert_shared(source,abi,tmp_path / "translation",optimizer=OPT,**options)
    capture,qualified = tmp_path / "capture",tmp_path / "qualified"
    write_effect_bundle(result,capture,arrays,reference(True),work=tmp_path / "native",python=NATIVE)
    qualify_kernel(capture,qualified,merlin_root=os.environ["T2M_MERLIN_ROOT"],python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=NATIVE,clang=os.environ["T2M_MERLIN_CLANG"])
    verify_kernel_qualification(qualified)
    features = json.loads((qualified / "host/compilation.json").read_text())["features"]
    assert feature in features and "lower_length_aware_diagnostics" in features
    assert (qualified / "evaluation/stdout.bin").read_bytes() == reference(True)["stdout"].encode("utf-8")
    if kind == "atomic":
        assert "OK integer_compare_exchange" in (capture / "native/compiler.log").read_text()
        receipt = verify_bundle(capture)
        assert "native/compiler.log" in receipt["artifacts"]
    signature = json.loads((qualified / "compiler/signature.json").read_text())
    assert len(signature["outputs"]) == 1
