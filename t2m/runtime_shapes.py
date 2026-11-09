"""Evaluate declared logical shape bindings for JIT and compiled host callers.

Pure Python so isolated workers can share this ABI mechanism without importing
the compiler frontend, NumPy or any Triton dependency.
"""


def resolve_shapes(contract, arrays):
    if contract.get("schema") not in ("t2m.runtime-shapes.v1", "merlin.runtime-shapes.v1") or len(contract.get("inputs", [])) != len(arrays):
        raise ValueError("runtime shape contract has an incomplete input signature")
    scalars = {}
    for declared, array in zip(contract["inputs"], arrays, strict=True):
        if "scalar" not in declared:
            continue
        if tuple(array.shape) != () or declared["shape"] != [] or declared["scalar"] in scalars:
            raise ValueError("runtime shape scalar binding is invalid")
        value = array.item()
        if "bounds" in declared and not declared["bounds"][0] <= value <= declared["bounds"][1]:
            raise ValueError("runtime shape scalar is outside its guarded ABI bounds")
        scalars[declared["scalar"]] = value

    def shape(declared):
        result = []
        for size in declared["shape"]:
            size = scalars.get(size) if isinstance(size, str) else size
            if type(size) is not int or size < 0:
                raise ValueError("runtime dimension lacks a nonnegative integer scalar binding")
            result.append(size)
        return tuple(result)

    for declared, array in zip(contract["inputs"], arrays, strict=True):
        if tuple(array.shape) != shape(declared):
            raise ValueError("runtime tensor shape differs from its scalar dimension binding")
    return tuple(shape(value) for value in contract["outputs"])
