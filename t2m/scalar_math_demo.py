"""A visible real-source precise-math package with independent IEEE references."""

from pathlib import Path

import numpy as np

from .dynamic_abi import GridExtent
from .effect_bundle import write_effect_bundle
from .examples.scalar_math_kernels import precise_values
from .examples.scalar_math_reference import inputs,reference
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI,TensorBuffer,convert_shared


def demo(out,*,optimizer=None,python=None,memory_profile="tensor",address_policy="structured",effect_profile="ordinary",f32_atomic_add=None):
    out = Path(out)
    if out.exists(): raise FileExistsError(out)
    work = out.with_name(out.name+".work");work.mkdir(parents=True,exist_ok=False)
    names = ["x","y","low","high","no_nan","all_nan","divided","root"]
    source = work / "source.ttir"
    source.write_text(capture_ttir(precise_values,{**{name:"*fp32" for name in names},"n":"i32","B":"constexpr"},{"B":8}))
    abi = TensorABI(precise_values.__name__,tuple(TensorBuffer(name,("n",),"read" if i < 4 else "write") for i,name in enumerate(names)),
        grid=(GridExtent("n",8),),runtime_scalars=(RuntimeScalar("n","i32",8,bounds=(0,127)),))
    result = convert_shared(source,abi,work / "translation",optimizer=optimizer,memory_profile=memory_profile,
        address_policy=address_policy,effect_profile=effect_profile,f32_atomic_add=f32_atomic_add)
    values = inputs("f32")
    expected = {"schema":"t2m.effect-reference.v2","termination":"return","stdout":"","stderr":"",
        "results":{"archive":"reference.npz","comparisons":[{"kind":"values","rtol":0,"atol":0},*[{"kind":"ieee"}]*3]}}
    write_effect_bundle(result,out,[*values,np.asarray(len(values[0]),np.int32)],expected,references=reference(values,"f32"),
        work=work / "native",python=python)
    return {"bundle":str(out),"native_numerical_check":"passed","result_count":4,
        "result_policies":expected["results"]["comparisons"],"consumer_execution":"not-performed"}
