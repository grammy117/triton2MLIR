"""Native capability selection without importing frontend dialect bindings."""


def needs_region_state_protection(operation, ir):
    """Probe actual native types/control; provenance supplies no selection."""
    def walk(parent):
        yield parent
        for region in parent.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from walk(child.operation)

    for child in walk(operation):
        if child.name not in ("scf.for", "scf.while", "linalg.generic", "linalg.reduce",
                              "linalg.fill", "tensor.insert", "tensor.insert_slice"):
            continue
        scope = child.parent
        while scope is not None and not scope.name.startswith("linalg."):
            scope = scope.parent
        if scope is None:
            continue
        for value in child.operands:
            if not isinstance(value.type, ir.RankedTensorType):
                continue
            # Canonicalization/CSE can merge currently local pure destinations
            # with an ancestor's. Select the native preparation from actual
            # tensor/control operations; the consumer checks exact captures.
            return True
    return False
