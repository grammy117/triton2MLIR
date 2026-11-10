"""Visible real-source effect examples; production lowering remains generic."""

from pathlib import Path

import numpy as np

from .dynamic_abi import GridExtent
from .effect_bundle import write_effect_bundle
from .examples.effect_kernels import check_scalar,check_tensor,constant_check,fail_before_store,check_diagnostic
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI,TensorBuffer,convert_shared


def demo(out,*,case="constant-fail",optimizer=None,python=None,memory_profile="tensor",address_policy="structured"):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name+".work")
    work.mkdir(parents=True,exist_ok=False)
    expected,references = None,None
    if case in ("constant-fail","empty-return"):
        fail = case == "constant-fail"
        kernel,signature,constants = constant_check,{"FAIL":"constexpr"},{"FAIL":fail}
        abi,arrays,message = TensorABI(kernel.__name__,(),effect_only=True,grid=(3,)),[],"constant source assertion"
    elif case in ("scalar-check","literal-check"):
        kernel,signature,constants = check_scalar,{"value":"i32","steps":"i32"},{}
        message = "effect scalar must be nonnegative"
        if case == "literal-check":
            kernel = check_diagnostic
            message = 'before\x00after %s "value"\nλ\x00end'
            signature = dict(signature,MESSAGE="constexpr")
            constants = {"MESSAGE":message}
        abi = TensorABI(kernel.__name__,(),runtime_scalars=(RuntimeScalar("value","i32",0,bounds=(-9,9)),
            RuntimeScalar("steps","i32",1,bounds=(0,4))),effect_only=True,grid=(3,))
        arrays = [np.asarray(-1,np.int32),np.asarray(3,np.int32)]
        fail = arrays[1].item() > 0 and arrays[0].item() < 0
    elif case == "tensor-check":
        kernel,signature,constants = check_tensor,{"x":"*i32","n":"i32","limit":"i32","B":"constexpr"},{"B":8}
        abi = TensorABI(kernel.__name__,(TensorBuffer("x",("n",),dtype="i32"),),grid=(GridExtent("n",8),),
            runtime_scalars=(RuntimeScalar("n","i32",1,bounds=(0,17)),RuntimeScalar("limit","i32",2,bounds=(-9,9))),effect_only=True)
        arrays = [np.arange(13,dtype=np.int32)%5,np.asarray(13,np.int32),np.asarray(3,np.int32)]
        fail,message = bool(np.any(arrays[0] > arrays[2])),"effect tile exceeds limit"
    elif case == "fail-before-store":
        kernel,signature,constants = fail_before_store,{"out":"*i32","B":"constexpr"},{"B":8}
        abi,arrays = TensorABI(kernel.__name__,(TensorBuffer("out",(8,),"write","i32"),)),[]
        fail,message = True,"failure precedes real output"
    elif case in ("print-words","print-hex"):
        from .examples.print_kernels import print_integer_words
        from .examples.print_reference import word_reference
        kernel = print_integer_words
        signature = {"x":"*i64","n":"i32","steps":"i32","allow":"i1","B":"constexpr","HEX":"constexpr"}
        constants = {"B":8,"HEX":case == "print-hex"}
        abi = TensorABI(kernel.__name__,(TensorBuffer("x",("n",),dtype="i64"),),grid=(GridExtent("n",8),),
            runtime_scalars=(RuntimeScalar("n","i32",1,bounds=(0,65)),RuntimeScalar("steps","i32",2,bounds=(0,3)),
                            RuntimeScalar("allow","i1",3)),effect_only=True)
        values = np.resize(np.asarray([-(2**63),-1,0,2**63-1,127,255,2**32+1,-7],np.int64),13)
        arrays = [values,np.asarray(13,np.int32),np.asarray(1,np.int32),np.asarray(True)]
        expected = word_reference(values,13,1,True,hex_=constants["HEX"])
    elif case == "mixed-results":
        from .examples.mixed_effect_kernels import update_and_print
        from .examples.mixed_effect_reference import update_reference
        kernel = update_and_print
        signature = {"x":"*i64","state":"*i64","out":"*i64","n":"i32","steps":"i32","allow":"i1","B":"constexpr"}
        constants = {"B":8}
        abi = TensorABI(kernel.__name__,(TensorBuffer("x",("n",),dtype="i64"),
            TensorBuffer("state",("n",),"readwrite","i64"),TensorBuffer("out",("n",),"write","i64")),grid=(GridExtent("n",8),),
            runtime_scalars=(RuntimeScalar("n","i32",3,bounds=(0,17)),RuntimeScalar("steps","i32",4,bounds=(0,3)),
                RuntimeScalar("allow","i1",5)))
        x,state = np.arange(13,dtype=np.int64)+2**53+1,np.arange(13,dtype=np.int64)*3-7
        arrays = [x,state,np.asarray(13,np.int32),np.asarray(3,np.int32),np.asarray(True)]
        expected,references = update_reference(x,state,13,3,True)
    else:
        raise ValueError("unknown effect demonstration")
    source = work / "kernel.ttir"
    source.write_text(capture_ttir(kernel,signature,constants,debug=True))
    result = convert_shared(source,abi,work / "translation",optimizer=optimizer,memory_profile=memory_profile,address_policy=address_policy)
    if expected is None:
        expected = {"schema":"t2m.effect-reference.v1","termination":"assertion_failure" if fail else "return",
            "stdout":"Assertion `"+message+"` failed\n" if fail else "","stderr":""}
    write_effect_bundle(result,out,arrays,expected,work=work / "native",python=python,references=references)
    return {"bundle":str(out),"reference_comparison":"passed","termination":expected["termination"],
        "declared_result_count":len(abi.outputs),"result_count":len(abi.outputs) if references is not None else 0,
        "consumer_execution":"not-performed"}
