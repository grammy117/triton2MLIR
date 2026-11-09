"""Exact dense-stride cancellation over upstream PtrState expressions.

Meta 17f0c234 PtrAnalysis::mulState scales per-axis element offsets and
strides together. Retain those decomposed axes and cancel a declared dense
stride algebraically. No runtime division or arithmetic decomposition is
introduced; only a proved affine logical coordinate is returned.
"""

from collections import Counter
from dataclasses import asdict, dataclass

from .diagnostics import ConversionError
from .shared_indexing import Linear, linear


@dataclass(frozen=True)
class Polynomial:
    terms: tuple

    @classmethod
    def make(cls, values):
        terms = {}
        for atoms, coefficient in values:
            atoms = tuple(sorted(atoms))
            terms[atoms] = terms.get(atoms, 0) + coefficient
        if len(terms) > 128:
            raise ConversionError("dense stride expression exceeds its symbolic term budget")
        return cls(tuple(sorted((a, c) for a, c in terms.items() if c)))

    @classmethod
    def constant(cls, value):
        return cls.make([((), value)])

    @classmethod
    def affine(cls, form):
        values = [((), form.constant)]
        values.extend(((("p", axis),), c) for axis, c in enumerate(form.programs) if c)
        values.extend((() if low == high else (("l", identity, low, high),), c * low if low == high else c)
                      for identity, low, high, c in form.loops)
        values.extend((() if low == high else (("v", name, low, high),), c * low if low == high else c)
                      for name, low, high, c in form.parameters)
        return cls.make(values)

    def __add__(self, other):
        return self.make((*self.terms, *other.terms))

    def __mul__(self, other):
        if len(self.terms) * len(other.terms) > 256:
            raise ConversionError("dense stride multiplication exceeds its symbolic term budget")
        return self.make(((*a, *b), x * y) for a, x in self.terms for b, y in other.terms)

    def scale(self, number):
        return self.make((a, c * number) for a, c in self.terms)

    def bounds(self, grid):
        grid = (*grid, *((1,) * (3 - len(grid))))
        low = high = 0
        for atoms, coefficient in self.terms:
            left = right = coefficient
            for atom in atoms:
                a, b = (0, grid[atom[1]] - 1) if atom[0] == "p" else atom[2:]
                products = (left * a, left * b, right * a, right * b)
                left, right = min(products), max(products)
            low, high = low + left, high + right
        return low, high

    def divide_stride(self, divisor):
        if len(divisor.terms) != 1 or divisor.terms[0][1] <= 0:
            raise ConversionError("dense stride needs a positive monomial layout proof")
        factors, scale = divisor.terms[0]
        required, values = Counter(factors), []
        for atoms, coefficient in self.terms:
            available = Counter(atoms)
            if coefficient % scale or any(available[a] < n for a, n in required.items()):
                raise ConversionError("element offset is not exactly aligned with its declared dense stride")
            remaining = available - required
            values.append((tuple(a for a in sorted(remaining) for _ in range(remaining[a])), coefficient // scale))
        return self.make(values).to_linear()

    def split_stride(self, divisor):
        """Partition a uniform origin by exact monomial factors, without division in IR."""
        if len(divisor.terms) != 1 or divisor.terms[0][1] <= 0:
            raise ConversionError("dense stride needs a positive monomial layout proof")
        factors, scale = divisor.terms[0]
        required, aligned, remainder = Counter(factors), [], []
        for atoms, coefficient in self.terms:
            available = Counter(atoms)
            if coefficient % scale or any(available[a] < n for a, n in required.items()):
                remainder.append((atoms, coefficient))
            else:
                remaining = available - required
                aligned.append((tuple(a for a in sorted(remaining) for _ in range(remaining[a])), coefficient // scale))
        return self.make(aligned), self.make(remainder)

    def to_linear(self):
        result = Linear()
        for atoms, coefficient in self.terms:
            if not atoms:
                term = Linear(coefficient)
            elif len(atoms) != 1:
                raise ConversionError("logical coordinates remain nonlinear after dense-stride cancellation")
            elif atoms[0][0] == "p":
                term = Linear(0, tuple(coefficient if i == atoms[0][1] else 0 for i in range(3)))
            elif atoms[0][0] == "l":
                term = Linear(loops=((*atoms[0][1:], coefficient),))
            elif atoms[0][0] == "v":
                term = Linear(parameters=((*atoms[0][1:], coefficient),))
            else:
                raise ConversionError("logical coordinate retains an uneliminated local index")
            result = result + term
        return result


def expression(value, facts, grid, cache=None):
    cache = {} if cache is None else cache
    if value in cache:
        if cache[value] is None:
            raise ConversionError("cyclic dense-stride expression")
        return cache[value]
    if len(cache) >= 128:
        raise ConversionError("dense stride expression exceeds its SSA analysis budget")
    cache[value] = None
    if value in facts:
        result = Polynomial.affine(facts[value])
    else:
        op = value.owner
        name = getattr(op, "name", "")
        if name in ("arith.index_cast", "arith.extsi", "arith.extui", "arith.trunci"):
            result = expression(op.operands[0], facts, grid, cache)
            if name == "arith.extui" and result.bounds(grid)[0] < 0:
                raise ConversionError("unsigned dense-stride extension needs a nonnegative source")
        elif name in ("arith.addi", "arith.subi", "arith.muli"):
            a, b = (expression(v, facts, grid, cache) for v in op.operands)
            result = a * b if name == "arith.muli" else a + b.scale(-1 if name == "arith.subi" else 1)
        else:
            result = Polynomial.affine(linear(value, facts, grid))
    width = getattr(getattr(value.type, "width", None), "data", 64)
    lo, hi = result.bounds(grid)
    if not -(1 << (width - 1)) <= lo <= hi < 1 << (width - 1):
        raise ConversionError("dense stride arithmetic or cast may overflow")
    cache[value] = result
    return result


def dense_strides(abi, buffer):
    strides, stride = [], Polynomial.constant(1)
    for dimension in reversed(buffer.shape):
        strides.insert(0, stride)
        form = abi.dimension(dimension)
        stride = stride * (Polynomial.affine(form) if isinstance(form, Linear) else Polynomial.constant(form))
    return tuple(strides)


def recover_axes(abi, buffer, sizes, strides, offsets, facts, *, evidence=None):
    if not len(sizes) == len(strides) == len(offsets) == len(buffer.shape):
        raise ConversionError("dynamic dense descriptor rank differs from its declared tensor")
    if len(sizes) > 8:
        raise ConversionError("dynamic dense axis recovery exceeds the eight-axis analysis budget")
    expected = dense_strides(abi, buffer)
    candidates, stride_candidates, values, observed_strides = [], [], [], []
    for size, stride, offset in zip(sizes, strides, offsets, strict=True):
        observed = stride if isinstance(stride, Polynomial) else expression(stride, facts, abi.grid_bounds) if not isinstance(stride, int) else Polynomial.constant(stride)
        value = offset if isinstance(offset, Polynomial) else expression(offset, facts, abi.grid_bounds) if not isinstance(offset, int) else Polynomial.constant(offset)
        values.append(value)
        observed_strides.append(observed)
        choices, stride_choices = [], []
        for axis, layout in enumerate(expected):
            if observed != layout and not (observed == Polynomial.constant(0) and size == 1):
                continue
            stride_choices.append(axis)
            try:
                coordinate = value.divide_stride(layout)
            except ConversionError:
                continue
            choices.append((axis, coordinate))
        candidates.append(choices)
        stride_candidates.append(stride_choices)
    solutions, attempts = [], [0]
    def solve(index, used, coordinates, axes):
        attempts[0] += 1
        if attempts[0] > 256:
            raise ConversionError("dynamic dense axis correspondence exceeds its bounded search")
        if len(solutions) > 1:
            return
        if index == len(candidates):
            solutions.append((tuple(coordinates), tuple(axes)))
            return
        for axis, coordinate in candidates[index]:
            if axis not in used:
                current = list(coordinates)
                current[axis] = coordinate
                solve(index + 1, used | {axis}, current, [*axes, axis])
    solve(0, set(), [None] * len(expected), [])
    if not solutions:
        # A singleton source axis has no lane variation. PtrAnalysis may put
        # several uniform address contributions there. Repartition their sum
        # only when the stride correspondence is unique, with exactly one
        # degenerate axis; retain independent bounds/mask/ownership gates.
        degenerate = [i for i, (size, stride) in enumerate(zip(sizes, observed_strides, strict=True))
                      if size == 1 and stride == Polynomial.constant(0)]
        if len(degenerate) == 1:
            candidates = [[(axis, Linear()) for axis in choices] for choices in stride_candidates]
            attempts[0] = 0
            solve(0, set(), [None] * len(expected), [])
            if len(solutions) == 1:
                total = sum(values, Polynomial.constant(0))
                remainder, coordinates = total, []
                for stride in expected[:-1]:
                    coordinate, remainder = remainder.split_stride(stride)
                    coordinates.append(coordinate.to_linear())
                coordinates.append(remainder.divide_stride(expected[-1]))
                reconstructed = sum((Polynomial.affine(c) * stride for c, stride in zip(coordinates, expected, strict=True)),
                                    Polynomial.constant(0))
                if reconstructed != total:
                    raise ConversionError("dense uniform origin reconstruction failed")
                axes = solutions[0][1]
                if evidence is not None:
                    evidence.append({"kind": "exact-dense-uniform-origin-partition", "buffer": buffer.name,
                                     "axes": list(axes), "coordinates": [asdict(c) for c in coordinates]})
                return tuple(coordinates), axes
        raise ConversionError("upstream dynamic offsets/strides do not match the declared dense logical axes")
    if len(solutions) > 1 and any(result != solutions[0] for result in solutions[1:]):
        raise ConversionError("dynamic dense axis correspondence is ambiguous")
    return solutions[0]


@dataclass(frozen=True)
class DenseAxes:
    axes: tuple[int, ...]
