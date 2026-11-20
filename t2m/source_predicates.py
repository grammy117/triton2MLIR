"""Fixed-width branch constraints using the existing checked word comparisons.

SCF.cpp's IfOp region successors at LLVM a47bddcc establish which condition
is true in each region. InferIntRangeInterfaceImpls.cpp and its arithmetic
tests retain signed/unsigned word predicates. This lexical reader reuses the
source-index adapter's comparison sets; it is not a native path-range solver.
"""

from xdsl.dialects import arith, builtin

from .source_index_ranges import _allowed, _constant

INVERSE = {"eq": "ne", "ne": "eq", "slt": "sge", "sle": "sgt", "sgt": "sle", "sge": "slt",
           "ult": "uge", "ule": "ugt", "ugt": "ule", "uge": "ult"}
REVERSE = {"eq": "eq", "ne": "ne", "slt": "sgt", "sle": "sge", "sgt": "slt", "sge": "sle",
           "ult": "ugt", "ule": "uge", "ugt": "ult", "uge": "ule"}


def signed_comparison_enclosure(interval, predicate, word, width):
    """Intersect a signed source interval with an actual word comparison.

    Preserve sign-bit partitions for unsigned predicates. A disconnected set
    gets its conservative hull, so `x != 0` over both signs cannot remove an
    interior zero from the interval domain. None denotes an empty intersection.
    """
    if predicate not in INVERSE or not 1 <= width <= 4096:
        raise ValueError("source comparison needs a supported fixed-width predicate")
    modulus, half = 1 << width, 1 << (width - 1)
    low, high = interval
    if not -half <= low <= high < half or not 0 <= word < modulus:
        raise ValueError("source comparison exceeds its declared word domain")
    pieces = []
    for left, right in _allowed(predicate, word, width):
        for start, end, offset in ((0, half - 1, 0), (half, modulus - 1, modulus)):
            a, b = max(left, start) - offset, min(right, end) - offset
            a, b = max(a, low), min(b, high)
            if a <= b:
                pieces.append((a, b))
    return (min(a for a, _ in pieces), max(b for _, b in pieces)) if pieces else None


def branch_constraints(condition, truth):
    """Read necessary scalar predicates, never a sufficient-condition guess.

    True AND and false OR imply both operands. Their opposite outcomes do not.
    XOR with an actual i1 literal supplies boolean negation. Unknown conditions
    grant no optional facts; traversal is bounded independently of source data.
    """
    pending, constraints, visits = [(condition, truth)], [], 0
    while pending:
        value, selected = pending.pop()
        visits += 1
        if visits > 4096:
            return []
        if value.type != builtin.i1:
            continue
        op = value.owner
        if isinstance(op, arith.CmpiOp):
            number = op.predicate.value.data
            if not 0 <= number < len(arith.CMPI_COMPARISON_OPERATIONS):
                continue
            predicate = arith.CMPI_COMPARISON_OPERATIONS[number]
            predicate = predicate if selected else INVERSE[predicate]
            for argument, bound, pred in ((*op.operands, predicate), (*reversed(op.operands), REVERSE[predicate])):
                if not isinstance(argument.type, builtin.IntegerType) or argument.type != bound.type:
                    continue
                width = argument.type.bitwidth
                word = _constant(bound)
                if 1 < width <= 4096 and word is not None:
                    constraints.append((argument, pred, word, width))
                    break
        elif isinstance(op, arith.AndIOp) and selected or isinstance(op, arith.OrIOp) and not selected:
            pending.extend((argument, selected) for argument in op.operands)
        elif isinstance(op, arith.XOrIOp):
            for argument, literal in (op.operands, tuple(reversed(op.operands))):
                word = _constant(literal)
                if word in (0, 1):
                    pending.append((argument, selected != bool(word)))
                    break
    return constraints
