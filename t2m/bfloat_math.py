"""Detect scalar BF16 math; Merlin independently decides its legal forms."""


def requires_emulation(operations):
    transport = {"arith.constant","arith.bitcast","arith.extf","arith.truncf","arith.select"}
    return any((op.name.startswith("math.") or op.name.startswith("arith.") and op.name not in transport)
        and any(str(value.type) == "bf16" for value in (*op.operands,*op.results)) for op in operations)
