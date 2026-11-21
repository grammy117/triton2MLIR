"""Checked scalar recurrence structure and loop-local invariant values.

LLVM a47bddcc's LICM uses outside-region definitions and pure dependency
closures. Reuse that distinction without moving or speculating computation.
An outside load is an immutable SSA sample; a load in the loop is not invariant.
"""

from xdsl.dialects.builtin import IndexType, IntegerType
from xdsl.ir import BlockArgument

from .diagnostics import ConversionError

SCALAR_ARITH = frozenset({"arith.constant", "arith.addi", "arith.subi", "arith.muli",
    "arith.divsi", "arith.divui", "arith.ceildivsi", "arith.floordivsi", "arith.remsi", "arith.remui",
    "arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui", "arith.trunci", "arith.bitcast",
    "arith.minsi", "arith.maxsi", "arith.minui", "arith.maxui", "arith.andi", "arith.ori", "arith.xori",
    "arith.shli", "arith.shrsi", "arith.shrui", "arith.cmpi", "arith.select"})


def is_loop_invariant(value, loop, opcode):
    pending, visited = [value], set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if len(visited) > 256 or not isinstance(current.type, (IntegerType, IndexType)):
            return False
        if not loop.is_ancestor(current.owner):
            continue
        if isinstance(current, BlockArgument):
            return False
        op = current.owner
        if op.regions or opcode(op) not in SCALAR_ARITH:
            return False
        pending.extend(op.operands)
    return True


def additive_increment(loop, position, opcode):
    """Read actual additive edges; callers must separately bound each delta."""
    argument = loop.body.block.args[position + 1]
    if not isinstance(argument.type, (IntegerType, IndexType)):
        return None
    result = loop.body.block.last_op.operands[position]
    if result is argument:
        return None, 0
    op = result.owner
    kind = opcode(op) if hasattr(op, "name") else None
    if kind not in {"arith.addi", "arith.subi"}:
        return None
    a, b = op.operands
    if a is argument:
        delta, sign = b, -1 if kind == "arith.subi" else 1
    elif b is argument and kind == "arith.addi":
        delta, sign = a, 1
    else:
        return None
    return delta, sign


def invariant_increment(loop, position, opcode):
    recurrence = additive_increment(loop, position, opcode)
    if recurrence is None:
        return None
    delta, _sign = recurrence
    return recurrence if delta is None or is_loop_invariant(delta, loop, opcode) else None


def recurrence_bounds(domain, initial, delta, *, width):
    """Enclose actual additive state and zero trips, even for varying deltas.

    Each executed increment must lie within delta. No invariance, exact count,
    congruence or summation formula is inferred from the interval enclosure.
    """
    def after(counts):
        products = [n * value for n in counts for value in delta]
        return initial[0] + min(products), initial[1] + max(products)

    body = after((0, max(0, domain.maximum_iterations - 1)))
    final = after((domain.minimum_iterations, domain.maximum_iterations))
    if any(not -(1 << (width-1)) <= lo <= hi < 1 << (width-1) for lo, hi in (body, final)):
        raise ConversionError("source additive recurrence may overflow its carried width", opcode="scf.for")
    return body, final
