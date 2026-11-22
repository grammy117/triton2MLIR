"""Select FP8 numeric legalization from actual types; Merlin checks legality."""

FORMATS = frozenset({"f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ"})


def requires_emulation(operations):
    transport = {"arith.constant", "arith.bitcast", "arith.extf", "arith.truncf", "arith.select"}

    def element(type_):
        if hasattr(type_, "get_element_type"):
            return type_.get_element_type()
        return getattr(type_, "element_type", type_)

    return any((op.name.startswith("math.") or op.name.startswith("arith.") and op.name not in transport)
               and any(str(element(value.type)) in FORMATS for value in (*op.operands, *op.results))
               for op in operations)


def requires_widening(operations):
    def element(type_):
        if hasattr(type_, "get_element_type"):
            return type_.get_element_type()
        return getattr(type_, "element_type", type_)

    return any(
        op.name == "arith.extf"
        and any(str(element(value.type)) in FORMATS for value in op.operands)
        for op in operations
    )


def requires_narrowing(operations):
    def element(type_):
        if hasattr(type_, "get_element_type"):
            return type_.get_element_type()
        return getattr(type_, "element_type", type_)

    return any(
        op.name == "arith.truncf"
        and any(str(element(value.type)) in FORMATS for value in op.results)
        for op in operations
    )
