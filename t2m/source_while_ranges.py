"""Full fixed-width enclosures at actual SCF while interfaces.

LLVM a47bddcc IntegerRangeAnalysis.cpp uses the maximum storage-bit range
when a loop-carried lattice cannot establish a narrower invariant. Reuse that
sound boundary without running a fixed point: actual integer SSA ports stay
words, not mathematical induction variables. This grants no trip, address,
initialization, ownership or no-overflow fact for other source arithmetic.
"""

from xdsl.dialects import builtin

from .diagnostics import ConversionError


def while_word_interval(value, loop, opcode):
    if loop.name != "scf.while" or len(loop.regions) != 2 or any(len(region.blocks) != 1 for region in loop.regions):
        raise ConversionError("source while word state needs two actual single-block regions", opcode="scf.while")
    before, after = (region.block for region in loop.regions)
    condition, yielded = before.last_op, after.last_op
    if (condition is None or opcode(condition) != "scf.condition" or not condition.operands
            or condition.operands[0].type != builtin.i1
            or yielded is None or opcode(yielded) != "scf.yield"
            or tuple(v.type for v in before.args) != tuple(v.type for v in loop.operands)
            or tuple(v.type for v in yielded.operands) != tuple(v.type for v in loop.operands)
            or tuple(v.type for v in after.args) != tuple(v.type for v in loop.results)
            or tuple(v.type for v in condition.operands[1:]) != tuple(v.type for v in loop.results)
            or value not in (*before.args, *after.args, *loop.results)):
        raise ConversionError("source while word state needs complete typed condition/backedge interfaces", opcode="scf.while")
    scalar = value.type.element_type if isinstance(value.type, builtin.TensorType) else value.type
    if not isinstance(scalar, builtin.IntegerType) or not 1 <= scalar.bitwidth <= 4096:
        raise ConversionError("source while state needs a fixed-width integer word", opcode="scf.while")
    half = 1 << (scalar.bitwidth - 1)
    return -half, half - 1
