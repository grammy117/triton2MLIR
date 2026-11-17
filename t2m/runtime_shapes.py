"""Evaluate declared logical shape bindings for JIT and compiled host callers.

Pure Python so isolated workers can share this ABI mechanism without importing
the compiler frontend, NumPy or any Triton dependency.
"""


def resolve_shapes(contract, arrays):
    schema = contract.get("schema")
    axis_profile = schema in ("t2m.runtime-shapes.v2", "merlin.runtime-shapes.v2")
    if (schema not in ("t2m.runtime-shapes.v1", "merlin.runtime-shapes.v1") and not axis_profile
            or len(contract.get("inputs", [])) != len(arrays)):
        raise ValueError("runtime shape contract has an incomplete input signature")
    if not axis_profile and "dimension_bindings" in contract:
        raise ValueError("input-axis dimensions require runtime shape contract v2")
    if axis_profile and (set(contract) != {"schema", "inputs", "outputs", "dimension_bindings"}
            or not isinstance(contract["inputs"], list) or not isinstance(contract["outputs"], list)
            or any(not isinstance(value, dict) or not isinstance(value.get("shape"), list)
                   or "scalar" in value and (not isinstance(value["scalar"], str) or not value["scalar"].isidentifier())
                   for value in [*contract["inputs"], *contract["outputs"]])):
        raise ValueError("runtime shape v2 requires a complete explicit tensor/scalar signature")
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

    if axis_profile:
        bindings = contract.get("dimension_bindings")
        if not isinstance(bindings, dict):
            raise ValueError("runtime shape v2 requires explicit input-axis dimension bindings")
        for name, binding in bindings.items():
            if (not isinstance(name, str) or not name.isidentifier() or name in scalars
                    or not isinstance(binding, dict) or set(binding) not in (
                        {"argument", "axis"}, {"argument", "axis", "bounds"})):
                raise ValueError("runtime input-axis dimension binding is incomplete or ambiguous")
            argument, axis = binding["argument"], binding["axis"]
            if (type(argument) is not int or not 0 <= argument < len(arrays) or type(axis) is not int
                    or not 0 <= axis < len(arrays[argument].shape)
                    or axis >= len(contract["inputs"][argument]["shape"])
                    or "scalar" in contract["inputs"][argument]
                    or contract["inputs"][argument]["shape"][axis] != name):
                raise ValueError("runtime dimension must name its actual declared input axis")
            value = arrays[argument].shape[axis]
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError("runtime input dimension must fit its nonnegative i64 size domain")
            interval = binding.get("bounds")
            if "bounds" in binding and (not isinstance(interval, list) or len(interval) != 2
                    or any(type(item) is not int for item in interval)
                    or not 0 <= interval[0] <= interval[1] < 2**63):
                raise ValueError("runtime input dimension bounds must be ordered nonnegative i64 sizes")
            if "bounds" in binding and not interval[0] <= value <= interval[1]:
                raise ValueError("runtime input dimension is outside its declared bounds")
            scalars[name] = value

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
