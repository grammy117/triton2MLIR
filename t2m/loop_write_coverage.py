"""Exact scalar-store coverage of rectangular source loop nests.

Reuse the existing bounded polynomial reader over native pointer/index SSA.
Recover a mixed-radix bijection, independently of traversal order, without
enumerating source elements or changing the executable loops and memory.
"""

from xdsl.dialects import tensor

from .diagnostics import ConversionError
from .loop_domains import require_signed_loop
from .shared_indexing import Linear
from .symbolic_layout import Polynomial, expression

ZERO, ONE = Polynomial.constant(0), Polynomial.constant(1)


def substitute(poly, known):
    terms = []
    for atoms, coefficient in poly.terms:
        remaining = []
        for atom in atoms:
            if atom in known:
                coefficient *= known[atom]
            else:
                remaining.append(atom)
        terms.append((tuple(remaining), coefficient))
    return Polynomial.make(terms)


def rectangular_bijection(domains, index, *, grid):
    """Prove flat ordinal coverage when every rectangular bound is positive.

    The caller separately proves that the raw bound product equals the actual
    result size. Consequently any zero bound makes the result empty. Unit-bound
    facts below are local to a reachable store, not entry-domain restrictions.
    """
    known = {}
    for induction, upper in domains:
        low, high = upper.bounds(grid)
        if low < 0 or high < 1 or any(atom[0] == "l" for atoms, _ in upper.terms for atom in atoms):
            return None
        if induction == ZERO:
            if high > 1:
                return None
            # A nonnegative bound <= 1 equals 1 whenever its body executes.
            constant = sum(c for atoms, c in upper.terms if not atoms)
            varying = [(atoms, c) for atoms, c in upper.terms if atoms]
            inferred = {}
            if not varying:
                if constant != 1:
                    return None
            elif len(varying) == 1 and len(varying[0][0]) == 1:
                atoms, coefficient = varying[0]
                if (1 - constant) % coefficient:
                    return None
                inferred[atoms[0]] = (1 - constant) // coefficient
            elif not constant and len(varying) == 1 and varying[0][1] == 1 and all(
                    atom[0] == "v" and atom[2] >= 0 for atom in varying[0][0]):
                inferred = dict.fromkeys(varying[0][0], 1)
            else:
                return None
            if any(atom in known and known[atom] != value for atom, value in inferred.items()):
                return None
            known.update(inferred)

    digits, unit = {}, []
    for position, (induction, upper) in enumerate(domains):
        upper = substitute(upper, known)
        if induction == ZERO:
            unit.append(position)
            continue
        if len(induction.terms) != 1 or induction.terms[0][1] != 1 or len(induction.terms[0][0]) != 1:
            return None
        atom = induction.terms[0][0][0]
        if len(atom) != 4 or atom[0] != "l" or atom[2] != 0:
            return None
        if upper == ONE:
            known[atom] = 0
            unit.append(position)
        else:
            if atom in digits:
                return None
            digits[atom] = (position, upper)
    actual = substitute(index, known)
    coefficients = {atom: ZERO for atom in digits}
    for atoms, coefficient in actual.terms:
        selected = [atom for atom in atoms if atom in digits]
        if len(selected) != 1:
            return None  # No origin shift or cross-product of induction values.
        atom = selected[0]
        other = list(atoms)
        other.remove(atom)
        coefficients[atom] += Polynomial.make([(tuple(other), coefficient)])
    stride, order = ONE, []
    while digits:
        selected = next((atom for atom in digits if coefficients[atom] == stride), None)
        if selected is None:
            return None
        position, upper = digits.pop(selected)
        order.append(position)
        stride *= upper
    return {"memory_digit_order": order, "unit_loops": unit}


def scalar_stores(loop, abi, buffers, mapping, facts, opcode, *, outer_bounds=False, booleans=None, evidence=None):
    """Scalar addresses written on every Boolean path of a rectangular nest."""
    pending, groups = [(loop, (), 1)], {}
    while pending:
        op, domains, guard = pending.pop()
        if guard == 0:
            continue
        name = opcode(op)
        if name == "scf.for":
            try:
                require_signed_loop(op)
                lower, upper, step = (expression(mapping[value], facts, abi.grid_bounds) for value in op.operands[:3])
                if lower != ZERO or step != ONE or upper.bounds(abi.grid_bounds)[0] < 0 or not outer_bounds and any(
                        atom[0] == "l" for atoms, _ in upper.terms for atom in atoms):
                    continue
                induction = expression(mapping[op.body.block.args[0]], facts, abi.grid_bounds)
            except ConversionError:
                continue
            pending.extend((child, (*domains, (induction, upper)), guard) for child in reversed(tuple(op.body.block.ops)))
            continue
        if name == "scf.if" and booleans is not None:
            condition = booleans.condition(mapping.get(op.operands[0]))
            for region, selected in zip(op.regions, (condition, booleans.negate(condition)), strict=True):
                if len(region.blocks) == 1:
                    pending.extend((child, domains, booleans.combine("and", guard, selected))
                                   for child in reversed(tuple(region.block.ops)))
            continue
        if name != "tts.scatter" or len(op.operands) not in (3, 4) or op.operands[0] not in buffers:
            continue
        buffer = buffers[op.operands[0]]
        if buffer.effect != "write":
            continue
        if len(op.operands) == 4:
            if booleans is None:
                continue
            guard = booleans.combine("and", guard, booleans.condition(mapping[op.operands[3]]))
        index = mapping[op.operands[1]]
        if not isinstance(index.owner, tensor.FromElementsOp) or len(index.owner.operands) != 1:
            continue
        try:
            actual = expression(index.owner.operands[0], facts, abi.grid_bounds)
        except ConversionError:
            continue
        key = buffer.name, domains, actual
        if key not in groups:
            groups[key] = buffer, [], False
        saved, guards, conditional = groups[key]
        guards.append(guard)
        groups[key] = saved, guards, conditional or guard != 1
    for (_name, domains, actual), (buffer, guards, conditional) in groups.items():
        if (booleans.covers(guards) if booleans is not None else 1 in guards):
            if evidence is not None:
                evidence.clear()
                if conditional:
                    evidence["condition_coverage"] = "actual-source-boolean-partition"
            yield buffer, domains, actual


def complete_loop_writes(loop, abi, buffers, mapping, facts, opcode, *, evidence=None, booleans=None):
    """Prove actual unconditional stores cover the complete result allocation."""
    complete, conditions = {}, {}
    for buffer, domains, actual in scalar_stores(loop, abi, buffers, mapping, facts, opcode,
                                                booleans=booleans, evidence=conditions):
        if buffer.name in complete:
            continue
        count, expected = ONE, ONE
        for _iv, bound in domains:
            count *= bound
        for dimension in buffer.shape:
            value = abi.dimension(dimension)
            expected *= Polynomial.affine(value) if isinstance(value, Linear) else Polynomial.constant(value)
        if count != expected:
            continue
        proof = rectangular_bijection(domains, actual, grid=abi.grid_bounds)
        if proof is not None:
            complete[buffer.name] = buffer
            if evidence is not None:
                evidence[buffer.name] = {"coverage": "serial-contiguous-source-loop" if len(domains) == 1 else
                    "serial-rectangular-source-loop-nest", "nest_depth": len(domains),
                    "extent_equality": "actual-loop-upper-product-equals-result-element-count", **proof, **conditions}
    return list(complete.values())
