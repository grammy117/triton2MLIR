"""Select FP8 casts from actual types; Merlin decides complete legality."""

FORMATS = frozenset({"f8E4M3FN", "f8E4M3FNUZ", "f8E5M2", "f8E5M2FNUZ"})


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
