"""A tiled dot with source allocation spans and signed runtime memory strides."""

from pathlib import Path

from .dynamic_abi import GridExtent
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer, convert_shared


def convert(work, *, dtype="f32", optimizer=None):
    from .examples.strided_memory_kernels import strided_tiled_dot

    if dtype not in ("f32", "f16", "bf16"):
        raise ValueError("strided dot requires f32, f16 or bf16 inputs")
    work = Path(work)
    work.mkdir(parents=True,exist_ok=False)
    source = work / "source.ttir"
    names = strided_tiled_dot.arg_names
    signature = {name:"*" + {"f32":"fp32","f16":"fp16","bf16":"bf16"}[dtype] for name in names[:2]}
    signature.update(initial="*fp32",out="*fp32",m="i32",n="i32",k="i32",
                     **{name:"i64" for name in names[7:13]},a_count="i32",b_count="i32",
                     BM="constexpr",BN="constexpr",BK="constexpr")
    source.write_text(capture_ttir(strided_tiled_dot,signature,{"BM":16,"BN":16,"BK":32}))
    scalars = tuple(RuntimeScalar(name,signature[name],i,bounds={
        "m":(0,35),"n":(0,21),"k":(0,65),"a_count":(0,8192),"b_count":(0,8192),
        "a_base":(0,8191),"b_base":(0,8191)}.get(name,(-256,256))) for i,name in enumerate(names[4:15],4))
    abi = TensorABI("strided_tiled_dot",(TensorBuffer("x",("a_count",),dtype=dtype),
        TensorBuffer("y",("b_count",),dtype=dtype),TensorBuffer("initial",("m","n")),
        TensorBuffer("out",("m","n"),"write")),grid=(GridExtent("m",16),GridExtent("n",16)),runtime_scalars=scalars,
        intent="Tiled contraction over source allocations with runtime signed/padded/broadcast views; source K reuse; hardware mapping undecided")
    return convert_shared(source,abi,work / "translate",optimizer=optimizer,
                          memory_profile="guarded-buffer",address_policy="source-words")


def demo(out, *, dtype="f32", optimizer=None, python=None,
         memory_profile="guarded-buffer", address_policy="source-words"):
    from .effect_bundle import write_effect_bundle
    from .examples.strided_memory_reference import arguments,policy

    if memory_profile != "guarded-buffer" or address_policy != "source-words":
        raise ValueError("strided source-allocation demo requires guarded-buffer memory and source-words addresses")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    result = convert(work,dtype=dtype,optimizer=optimizer)
    values,expected = arguments(17,19,37,(-79,-2,-64,3),dtype)
    write_effect_bundle(result,out,values,policy(),references=[expected],work=work / "native",python=python)
    return {"bundle":str(out),"kernel":result.spec.entry,"input_dtype":dtype,
            "memory_profile":"guarded source allocation words","source_strides":[-79,-2,-64,3],
            "tiles":[16,16,32],"native_execution":"passed",
            "merlin_execution":"requires separate qualify-merlin invocation"}
