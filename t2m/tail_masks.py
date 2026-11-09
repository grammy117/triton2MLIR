"""Prove upstream rectangular tail extents without enumerating programs.

The shared MaskAnalysis (Meta 17f0c234, lib/Analysis/MaskAnalysis.cpp)
represents ``index < limit`` as max(min(start + tile, limit), start) - start,
sometimes with an additional min(tile, ...). Normalize those signed index
expressions structurally; accept only an established clipped affine prefix.
This adapts the algebraic contract, not upstream source code.
"""

from dataclasses import asdict, dataclass, field

from .diagnostics import ConversionError
from .shared_indexing import Linear, linear


@dataclass(frozen=True)
class TailExtent:
    cap: int
    limit: int | Linear
    offset: Linear
    value: object = field(compare=False, repr=False)

    def fits(self, offset, dimension):
        if isinstance(self.limit, Linear):
            # Limits recovered below contain guarded ABI symbols, never grid
            # coordinates/loop IVs. Their bounds therefore need no grid here.
            if any(self.limit.programs) or self.limit.loops:
                return False
            low, high = self.limit.bounds((1,))
        else:
            low = high = self.limit
        # With the separately proved nonnegative address offset, negative
        # limits select an empty prefix and make no memory observation.
        return self.offset == offset and low <= high <= dimension

    def as_dict(self):
        return {"maximum": self.cap, "limit": asdict(self.limit) if isinstance(self.limit, Linear) else self.limit, "offset": asdict(self.offset),
                "formula": "max(0,min(maximum,limit-offset))"}


def maximum(size):
    return size.cap if isinstance(size, TailExtent) else size


def describe(size):
    return size.as_dict() if isinstance(size, TailExtent) else size


def recover_extent(value, cap, offsets, facts, grid):
    """Match an exact clamp normal form; unsupported masks remain refusals."""
    cache = {}

    def bounds(expr):
        if isinstance(expr, Linear):
            return expr.bounds(grid)
        kind, children = expr
        ranges = [bounds(child) for child in children]
        choose = min if kind == "min" else max
        return choose(lo for lo, _ in ranges), choose(hi for _, hi in ranges)

    def combine(kind, *expressions):
        children = []
        for expr in expressions:
            values = expr[1] if not isinstance(expr, Linear) and expr[0] == kind else (expr,)
            for child in values:
                if child not in children:
                    children.append(child)
        retained = []
        for i, child in enumerate(children):
            redundant = False
            for j, other in enumerate(children):
                if i == j:
                    continue
                if isinstance(child, Linear) and isinstance(other, Linear):
                    lo, hi = (child + other.scale(-1)).bounds(grid)
                else:
                    a, b = bounds(child), bounds(other)
                    lo, hi = a[0] - b[1], a[1] - b[0]
                # For equal expressions retain the first, deterministically.
                if (lo >= 0 if kind == "min" else hi <= 0) and not (lo == hi == 0 and i < j):
                    redundant = True
                    break
            if not redundant:
                retained.append(child)
        if len(retained) == 1:
            return retained[0]
        return kind, tuple(sorted(retained, key=repr))

    def shift(expr, amount, sign=1):
        if isinstance(expr, Linear):
            return expr.scale(sign) + amount
        kind, children = expr
        if sign == -1:
            kind = "max" if kind == "min" else "min"
        return combine(kind, *(shift(child, amount, sign) for child in children))

    def expression(v):
        if v in cache:
            if cache[v] is None:
                raise ConversionError("cyclic tail mask expression")
            return cache[v]
        if len(cache) >= 128:
            raise ConversionError("tail mask expression exceeds its structural analysis budget")
        cache[v] = None  # Also bound recursion through malformed cyclic IR.
        try:
            result = linear(v, facts, grid)
        except ConversionError:
            op = v.owner
            name = getattr(op, "name", "")
            if name in ("arith.minsi", "arith.maxsi"):
                result = combine("min" if name == "arith.minsi" else "max",
                                 *(expression(operand) for operand in op.operands))
            elif name == "arith.select":
                predicate = v.owner.operands[0].owner
                if (getattr(predicate, "name", "") != "arith.cmpi" or predicate.predicate.value.data != 2
                        or expression(op.operands[1]) != Linear(1) or expression(op.operands[2]) != Linear(0)):
                    raise ConversionError("mask select is not a signed unit-prefix comparison", opcode=name)
                left, right = (linear(operand, facts, grid) for operand in predicate.operands)
                # For integral indices, (start < limit ? 1 : 0) is exactly
                # max(0,min(1,limit-start)); upstream uses this for unit tiles.
                result = combine("max", Linear(), combine("min", Linear(1), right + left.scale(-1)))
            elif name in ("arith.addi", "arith.subi"):
                left, right = (expression(operand) for operand in op.operands)
                sign = -1 if name == "arith.subi" else 1
                if isinstance(right, Linear):
                    result = shift(left, right.scale(sign))
                elif isinstance(left, Linear):
                    result = shift(right, left, sign)
                else:
                    raise ConversionError("mask combines two nonaffine extents", opcode=name)
            else:
                raise
        lo, hi = bounds(result)
        width = getattr(getattr(v.type, "width", None), "data", 64)
        if not -(1 << (width - 1)) <= lo <= hi < 1 << (width - 1):
            raise ConversionError("tail mask arithmetic may overflow")
        cache[v] = result
        return result

    tree = expression(value)
    if isinstance(tree, Linear) and not tree.dynamic:
        return tree.constant

    def leaves(expr):
        if isinstance(expr, Linear):
            yield expr
        else:
            for child in expr[1]:
                yield from leaves(child)

    for offset in offsets:
        for leaf in leaves(tree):
            limit = leaf + offset
            if any(limit.programs) or limit.loops:
                continue
            expected = combine("max", Linear(), combine("min", Linear(cap),
                               limit + offset.scale(-1)))
            if tree == expected:
                return TailExtent(cap, limit if limit.parameters else limit.constant, offset, value)
    raise ConversionError("mask is not a proved clipped affine prefix")
