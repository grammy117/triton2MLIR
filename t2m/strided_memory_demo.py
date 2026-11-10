"""A tiled dot with source allocation spans and signed runtime memory strides."""

from pathlib import Path

from .dynamic_abi import GridExtent
from .frontend import capture_ttir
from .scalar_abi import RuntimeScalar
from .shared_tensor import TensorABI, TensorBuffer, convert_shared


def capture(work, *, dtype="f32", writable=False):
    from .examples.strided_memory_kernels import strided_tiled_dot, strided_output_dot

    if dtype not in ("f32", "f16", "bf16"):
        raise ValueError("strided dot requires f32, f16 or bf16 inputs")
    work = Path(work)
    work.mkdir(parents=True,exist_ok=False)
    source = work / "source.ttir"
    kernel = strided_output_dot if writable else strided_tiled_dot
    names = kernel.arg_names
    signature = {name:"*" + {"f32":"fp32","f16":"fp16","bf16":"bf16"}[dtype] for name in names[:2]}
    signature.update(initial="*fp32",out="*fp32",m="i32",n="i32",k="i32",
                     **{name:"i64" for name in names[7:13]},a_count="i32",b_count="i32",
                     BM="constexpr",BN="constexpr",BK="constexpr")
    if writable:
        signature.update(out_row="i64",out_col="i64",out_base="i64",out_count="i32")
    source.write_text(capture_ttir(kernel,signature,{"BM":16,"BN":16,"BK":32}))
    scalars = tuple(RuntimeScalar(name,signature[name],i,bounds={
        "m":(0,35),"n":(0,21),"k":(0,65),"a_count":(0,8192),"b_count":(0,8192),
        "a_base":(0,8191),"b_base":(0,8191),"out_base":(0,8191),"out_count":(0,8192)
        }.get(name,(-256,256))) for i,name in enumerate(names[4:names.index("BM")],4))
    abi = TensorABI(kernel.__name__,(TensorBuffer("x",("a_count",),dtype=dtype),
        TensorBuffer("y",("b_count",),dtype=dtype),TensorBuffer("initial",("m","n")),
        TensorBuffer("out",("out_count",) if writable else ("m","n"),"readwrite" if writable else "write")),
        grid=(GridExtent("m",16),GridExtent("n",16)),runtime_scalars=scalars,
        intent=("Tiled contraction over source allocations with runtime signed/padded/broadcast views; source K reuse; hardware mapping undecided"
                + ("; partial strided output writes preserve incoming allocation state" if writable else "")))
    return source,abi


def convert(work, *, dtype="f32", optimizer=None, writable=False):
    source,abi = capture(work,dtype=dtype,writable=writable)
    return convert_shared(source,abi,Path(work) / "translate",optimizer=optimizer,
                          memory_profile="guarded-buffer",address_policy="source-words")


def demo(out, *, dtype="f32", optimizer=None, python=None,
         memory_profile="guarded-buffer", address_policy="source-words", writable=False):
    from .effect_bundle import write_effect_bundle
    from .examples.strided_memory_reference import arguments,output_arguments,policy

    if memory_profile != "guarded-buffer" or address_policy != "source-words":
        raise ValueError("strided source-allocation demo requires guarded-buffer memory and source-words addresses")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    result = convert(work,dtype=dtype,optimizer=optimizer,writable=writable)
    values,expected = (output_arguments(17,19,37,(-79,-2,-64,3),(-47,2),dtype) if writable
                       else arguments(17,19,37,(-79,-2,-64,3),dtype))
    write_effect_bundle(result,out,values,policy(words=writable),references=[expected],work=work / "native",python=python)
    return {"bundle":str(out),"kernel":result.spec.entry,"input_dtype":dtype,
            "memory_profile":"guarded source allocation words","source_strides":[-79,-2,-64,3],
            "tiles":[16,16,32],"output_storage":"incoming allocation with preserved holes" if writable else "dense fresh result",
            "native_execution":"passed",
            "merlin_execution":"requires separate qualify-merlin invocation"}
