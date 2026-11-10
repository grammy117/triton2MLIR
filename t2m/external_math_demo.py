"""Default external math declarations through the real Merlin package path."""

from pathlib import Path

import numpy as np

from .dynamic_abi import GridExtent
from .effect_bundle import write_effect_bundle
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer, convert_shared


def convert(work, dtype="f32", *, optimizer=None, memory_profile="tensor", address_policy="structured"):
    from .examples.external_math_kernels import external_values

    if dtype not in ("f32", "f64"):
        raise ValueError("external math demo requires f32 or f64")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    names = external_values.arg_names
    position = names.index("n")
    source = work / "source.ttir"
    signature = {name: "*fp32" if dtype == "f32" else "*fp64" for name in names[:position]}
    signature.update(n="i32", steps="i32", enabled="i1", B="constexpr")
    source.write_text(capture_ttir(external_values, signature, {"B":8}))
    abi = TensorABI("external_values", tuple(TensorBuffer(name,("n",),"read" if i < 2 else "write",dtype)
                    for i,name in enumerate(names[:position])), grid=(GridExtent("n",8),),
                    runtime_scalars=(RuntimeScalar("n","i32",position,bounds=(0,127)),
                                     RuntimeScalar("steps","i32",position+1,bounds=(0,3)),
                                     RuntimeScalar("enabled","i1",position+2)))
    return convert_shared(source,abi,work / "translate",optimizer=optimizer,memory_profile=memory_profile,address_policy=address_policy)


def demo(out, *, optimizer=None, python=None, memory_profile="tensor", address_policy="structured", dtype="f32"):
    from .examples.external_math_reference import inputs, policy, reference

    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    result = convert(work,dtype,optimizer=optimizer,memory_profile=memory_profile,address_policy=address_policy)
    x,y = inputs(dtype)
    write_effect_bundle(result,out,[x,y,np.asarray(len(x),np.int32),np.asarray(2,np.int32),np.asarray(True)],policy(dtype),
                        references=reference(x,y,dtype,2),work=work / "native",python=python)
    return {"bundle":str(out),"kernel":result.spec.entry,"dtype":dtype,"result_count":26,
            "native_execution":"passed","merlin_execution":"requires separate qualify-merlin invocation"}
