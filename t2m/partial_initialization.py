"""Membership in actual completed scalar-store images, without tracker arrays.

Reuse the bounded polynomial source reader and fixed-width predicate analysis.
Facts describe the image of a rectangular source loop, not a whole allocation.
Lexically varying symbols expire at their loop edge. No IR is rewritten here.
"""

from dataclasses import dataclass

from xdsl.dialects import arith, builtin, tensor

from .diagnostics import ConversionError
from .loop_domains import require_signed_loop
from .loop_write_coverage import ONE, ZERO
from .source_predicates import branch_constraints, signed_comparison_enclosure
from .symbolic_layout import Polynomial, expression


def atoms(poly):
    return {atom for term, _coefficient in poly.terms for atom in term}


def scalar(value):
    if isinstance(value.type, builtin.TensorType):
        owner = value.owner
        if isinstance(owner, tensor.FromElementsOp) and len(owner.operands) == 1:
            return owner.operands[0]
        return None
    return value


@dataclass(frozen=True)
class WriteImage:
    origin: Polynomial
    axes: tuple  # (actual trip count, element stride), including zero/unit axes.

    def depends_on(self, identity):
        return any(atom[0] == "l" and atom[1] == identity
                   for poly in (self.origin, *(v for pair in self.axes for v in pair)) for atom in atoms(poly))


def store_image(domains, index, grid):
    """Split actual indices into a fixed origin and independent written digits."""
    digits, units = {}, []
    for induction, count in domains:
        if count.bounds(grid)[0] < 0:
            return None
        if induction == ZERO:
            if count.bounds(grid)[1] > 1:
                return None
            units.append((count, ZERO))
        elif (len(induction.terms) == 1 and induction.terms[0][1] == 1 and len(induction.terms[0][0]) == 1
              and induction.terms[0][0][0][0] == "l" and induction.terms[0][0][0][2] == 0):
            atom = induction.terms[0][0][0]
            if atom in digits:
                return None
            digits[atom] = count
        else:
            return None
    if any(atoms(count) & digits.keys() for count in (*digits.values(), *(count for count, _ in units))):
        return None
    origin, strides = ZERO, dict.fromkeys(digits, ZERO)
    for term, coefficient in index.terms:
        selected = [atom for atom in term if atom in digits]
        if len(selected) > 1:
            return None
        if not selected:
            origin += Polynomial.make([(term, coefficient)])
        else:
            remaining = list(term)
            remaining.remove(selected[0])
            strides[selected[0]] += Polynomial.make([(tuple(remaining), coefficient)])
    axes = [*units, *((count, strides[atom]) for atom, count in digits.items())]
    return WriteImage(origin, tuple(sorted(axes, key=lambda pair: (pair[1].terms, pair[0].terms))))


class ReadFacts:
    def __init__(self, grid):
        self.grid, self.limits, self.nonnegative = (*grid, *((1,) * (3 - len(grid)))), {}, set()
        self.unreachable = False

    def bounds(self, poly):
        low = high = 0
        for term, coefficient in poly.terms:
            left = right = coefficient
            for atom in term:
                a, b = self.limits.get(atom, (0, self.grid[atom[1]] - 1) if atom[0] == "p" else atom[2:])
                products = (left * a, left * b, right * a, right * b)
                left, right = min(products), max(products)
            low, high = low + left, high + right
        return low, high

    def refine(self, poly, low, high):
        if low > high:
            self.unreachable = True
            return
        constant = sum(c for term, c in poly.terms if not term)
        varying = [(term, c) for term, c in poly.terms if term]
        if len(varying) == 1 and len(varying[0][0]) == 1:
            (atom,), coefficient = varying[0]
            if coefficient < 0:
                low, high, constant, coefficient = -high, -low, -constant, -coefficient
            lower, upper = (low - constant + coefficient - 1) // coefficient, (high - constant) // coefficient
            a, b = self.limits.get(atom, (0, self.grid[atom[1]] - 1) if atom[0] == "p" else atom[2:])
            self.limits[atom] = max(a, lower), min(b, upper)
            if self.limits[atom][0] > self.limits[atom][1]:
                self.unreachable = True
        elif low > 0 and not constant and len(varying) == 1 and varying[0][1] > 0:
            term, _coefficient = varying[0]
            if all(self.bounds(Polynomial.make([((atom,), 1)]))[0] >= 0 for atom in term):
                for atom in term:
                    base = Polynomial.make([((atom,), 1)])
                    self.refine(base, 1, self.bounds(base)[1])

    def positive(self, poly):
        low, high = self.bounds(poly)
        self.refine(poly, max(1, low), high)
        self.nonnegative.add(poly + ONE.scale(-1))

    def ge_zero(self, poly):
        return self.bounds(poly)[0] >= 0 or poly in self.nonnegative

    def contains(self, image, query):
        if self.unreachable:
            return True
        axes = sorted(image.axes, key=lambda pair: max((len(term), abs(c).bit_length())
                      for term, c in pair[1].terms) if pair[1].terms else (0, 0), reverse=True)
        if any(not self.ge_zero(count + ONE.scale(-1)) for count, _stride in axes):
            return False
        remaining, budget = query + image.origin.scale(-1), [128]

        def solve(position, delta):
            budget[0] -= 1
            if budget[0] < 0:
                return False
            if position == len(axes):
                return self.bounds(delta) == (0, 0)
            count, stride = axes[position]
            if self.bounds(stride) == (0, 0) or self.bounds(count) == (1, 1):
                return solve(position + 1, delta)
            choices = [ZERO, ONE, count + ONE.scale(-1)]
            if len(stride.terms) == 1:
                factors, scale = stride.terms[0]
                if scale < 0:
                    scale = -scale
                    divisor = stride.scale(-1)
                else:
                    divisor = stride
                if scale:
                    quotient, _remainder = delta.split_stride(divisor)
                    # Constant terms also have an exact Euclidean quotient.
                    if not factors:
                        constant = sum(c for term, c in delta.terms if not term)
                        quotient += Polynomial.constant(constant // scale if constant % scale else 0)
                    choices.insert(0, quotient.scale(-1 if stride.terms[0][1] < 0 else 1))
            for coordinate in dict.fromkeys(choices):
                if (self.ge_zero(coordinate) and self.ge_zero(count + coordinate.scale(-1) + ONE.scale(-1))
                        and solve(position + 1, delta + (coordinate * stride).scale(-1))):
                    return True
            return False

        return solve(0, remaining)


def read_facts(source, mapping, facts, abi, mask):
    """Facts on the actual selected source read edge, preserving definition checks."""
    proof = ReadFacts(abi.grid_bounds)
    conditions = [] if mask is None else [(scalar(mask), True)]
    parent = source.parent_op()
    while parent is not None:
        if parent.name == "scf.if":
            child = source
            while child.parent_op() is not parent:
                child = child.parent_op()
            conditions.append((mapping.get(parent.operands[0]), child.parent_region() is parent.regions[0]))
        elif parent.name == "scf.for":
            try:
                require_signed_loop(parent)
                lower, upper, step = (expression(mapping[value], facts, abi.grid_bounds) for value in parent.operands[:3])
                induction = expression(mapping[parent.body.block.args[0]], facts, abi.grid_bounds)
                if lower == ZERO and step == ONE and upper.bounds(abi.grid_bounds)[0] >= 0:
                    proof.positive(upper)
                    proof.nonnegative.add(upper + induction.scale(-1) + ONE.scale(-1))
            except (ConversionError, KeyError):
                pass
        parent = parent.parent_op()
    for condition, truth in conditions:
        if condition is None:
            continue
        condition = scalar(condition)
        if condition is None:
            continue
        owner = condition.owner
        if (isinstance(owner, arith.ConstantOp) and isinstance(owner.value, builtin.IntegerAttr)
                and bool(owner.value.value.data) != truth):
            proof.unreachable = True
        for value, predicate, word, width in branch_constraints(condition, truth):
            try:
                poly = expression(value, facts, abi.grid_bounds)
                interval = signed_comparison_enclosure(proof.bounds(poly), predicate, word, width)
                if interval is None:
                    proof.unreachable = True
                else:
                    proof.refine(poly, *interval)
            except (ConversionError, ValueError):
                pass
    return proof


def initialized_read(source, index, mask, prefix, images, mapping, facts, abi):
    """Admit only a source-proved active scalar member of an actual write image."""
    if prefix is not None or not isinstance(index.type, builtin.TensorType) or any(n != 1 for n in index.type.get_shape()):
        return None
    value = scalar(index)
    if value is None:
        return None
    try:
        query = expression(value, facts, abi.grid_bounds)
        proof = read_facts(source, mapping, facts, abi, mask)
        if proof.unreachable:
            return {"coverage": "inactive-source-read", "storage": "no-initialization-tracker"}
        for image in images:
            if proof.contains(image, query):
                return {"coverage": "completed-source-store-image", "axes": len(image.axes),
                        "scope": "active-scalar-read-only", "storage": "no-initialization-tracker"}
    except ConversionError:
        pass
    return None
