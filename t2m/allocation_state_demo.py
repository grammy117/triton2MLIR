"""Tiled GEMM with accumulator and output pointers sharing allocation state."""

from dataclasses import replace
from pathlib import Path

from .allocation_bindings import PointerArgument, ScalarArgument, convert_allocations
from .scalar_abi import RuntimeScalar
from .strided_memory_demo import capture


def convert(work, *, dtype="f32", optimizer=None):
    work = Path(work)
    work.mkdir(parents=True,exist_ok=False)
    source,original = capture(work / "capture",dtype=dtype,writable=True)
    abi = replace(original,entry=original.entry+"_allocations",
        buffers=(*original.buffers[:2],replace(original.buffers[3],name="state")),
        runtime_scalars=(*(replace(scalar,source_arg=scalar.source_arg-1) for scalar in original.runtime_scalars),
            RuntimeScalar("c_origin","i64",18,bounds=(0,8191))),
        intent=original.intent+"; source accumulator/output pointers share one explicit owning allocation")
    bindings = (PointerArgument("x"),PointerArgument("y"),PointerArgument("state","c_origin"),
                PointerArgument("state"),*(ScalarArgument(scalar.name) for scalar in original.runtime_scalars))
    return convert_allocations(source,abi,bindings,work / "binding",source_entry=original.entry,
        optimizer=optimizer,memory_profile="guarded-buffer",address_policy="source-words")


def demo(out, *, dtype="f32", optimizer=None, python=None,
         memory_profile="guarded-buffer", address_policy="source-words"):
    from .effect_bundle import write_effect_bundle
    from .examples.allocation_state_reference import arguments
    from .examples.strided_memory_reference import policy

    if memory_profile != "guarded-buffer" or address_policy != "source-words":
        raise ValueError("allocation-state demo requires guarded-buffer memory and source-words addresses")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name+".work")
    result = convert(work,dtype=dtype,optimizer=optimizer)
    values,expected = arguments(7,11,37,(79,2,64,3),(-29,2),dtype)
    write_effect_bundle(result,out,values,policy(words=True),references=[expected],work=work / "native",python=python)
    return {"bundle":str(out),"kernel":result.spec.entry,"input_dtype":dtype,"tiles":[16,16,32],
        "source_pointer_binding":"C and output share one owning readwrite root; partial strided output",
        "native_execution":"passed","merlin_execution":"requires separate qualify-merlin invocation"}
