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


def complete_loop_writes(loop, abi, buffers, mapping, facts, opcode, *, evidence=None):
    """Prove actual unconditional stores cover the complete result allocation."""
    complete, pending = {}, [(loop, ())]
    while pending:
        current, ancestors = pending.pop()
        try:
            require_signed_loop(current)
            lower, upper, step = (expression(mapping[value], facts, abi.grid_bounds) for value in current.operands[:3])
            if lower != ZERO or step != ONE or upper.bounds(abi.grid_bounds)[0] < 0 or any(
                    atom[0] == "l" for atoms, _ in upper.terms for atom in atoms):
                continue
            induction = expression(mapping[current.body.block.args[0]], facts, abi.grid_bounds)
        except ConversionError:
            continue
        domains = (*ancestors, (induction, upper))
        count = ONE
        for _iv, bound in domains:
            count *= bound
        for op in current.body.block.ops:
            if opcode(op) == "scf.for":
                pending.append((op, domains))
                continue
            # Conditional/masked stores supply no unconditional write credit.
            if opcode(op) != "tts.scatter" or len(op.operands) != 3 or op.operands[0] not in buffers:
                continue
            buffer = buffers[op.operands[0]]
            if buffer.effect != "write" or buffer.name in complete:
                continue
            index = mapping[op.operands[1]]
            if not isinstance(index.owner, tensor.FromElementsOp) or len(index.owner.operands) != 1:
                continue
            try:
                actual = expression(index.owner.operands[0], facts, abi.grid_bounds)
                expected = ONE
                for dimension in buffer.shape:
                    value = abi.dimension(dimension)
                    expected *= Polynomial.affine(value) if isinstance(value, Linear) else Polynomial.constant(value)
                if count != expected:
                    continue
                proof = rectangular_bijection(domains, actual, grid=abi.grid_bounds)
            except ConversionError:
                continue
            if proof is not None:
                complete[buffer.name] = buffer
                if evidence is not None:
                    evidence[buffer.name] = {"coverage": "serial-contiguous-source-loop" if len(domains) == 1 else
                        "serial-rectangular-source-loop-nest", "nest_depth": len(domains),
                        "extent_equality": "actual-loop-upper-product-equals-result-element-count", **proof}
    return list(complete.values())
