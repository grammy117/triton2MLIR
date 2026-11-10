"""Package overlapping readonly operands alongside independent writable state."""

from pathlib import Path

from .strided_memory_demo import convert


def demo(out, *, dtype="f32", optimizer=None, python=None,
         memory_profile="guarded-buffer", address_policy="source-words"):
    from .effect_bundle import write_effect_bundle
    from .examples.readonly_alias_reference import aliased_arguments
    from .examples.strided_memory_reference import policy
    from .input_storage import READONLY_SNAPSHOTS

    if memory_profile != "guarded-buffer" or address_policy != "source-words":
        raise ValueError("readonly allocation demo requires guarded-buffer memory and source-words addresses")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    work = out.with_name(out.name + ".work")
    result = convert(work,dtype=dtype,optimizer=optimizer,writable=True)
    strides = (79,2,64,3)
    values,expected,owner = aliased_arguments(17,19,37,strides,dtype,writable=True)
    before = owner.tobytes()
    write_effect_bundle(result,out,values,policy(words=True),references=[expected],
                        work=work / "native",python=python)
    if owner.tobytes() != before:
        raise ValueError("readonly source allocation changed during packaging")
    return {"bundle":str(out),"kernel":result.spec.entry,"input_dtype":dtype,
            "input_preparation":READONLY_SNAPSHOTS,"source_strides":list(strides),
            "tiles":[16,16,32],"output_storage":"independent incoming allocation with preserved holes",
            "native_execution":"passed",
            "merlin_execution":"requires separate qualify-merlin invocation"}
