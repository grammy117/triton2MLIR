"""Bounded symbolic affine indices and rectangular predicates, without enumeration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Index:
    constant: int = 0
    program: int = 0
    lane: int = 0
    width: int = 0

    def combine(self, other, sign=1):
        if self.width and other.width and self.width != other.width:
            raise ValueError("incompatible index tile widths")
        return Index(self.constant + sign * other.constant, self.program + sign * other.program,
                     self.lane + sign * other.lane, max(self.width, other.width))

    def scale(self, factor):
        return Index(self.constant * factor, self.program * factor, self.lane * factor, self.width)

    def bounds(self, programs):
        terms = ((self.program, programs - 1), (self.lane, max(0, self.width - 1)))
        return (self.constant + sum(min(0, coefficient * bound) for coefficient, bound in terms),
                self.constant + sum(max(0, coefficient * bound) for coefficient, bound in terms))

    def uniform(self):
        if self.lane:
            raise ValueError("expected a block-uniform bound")
        return Index(self.constant, self.program)


def dominates(a, b, programs):
    """Whether a(p) >= b(p) over the entire declared logical grid."""
    return a.combine(b, -1).bounds(programs)[0] >= 0


@dataclass(frozen=True)
class Mask:
    lower: tuple[Index, ...] = ()
    upper: tuple[Index, ...] = ()

    def intersect(self, other):
        return Mask(tuple(dict.fromkeys(self.lower + other.lower)),
                    tuple(dict.fromkeys(self.upper + other.upper)))

    @classmethod
    def false(cls):
        return cls((Index(1),), (Index(0),))

    def empty(self, width, programs):
        return any(dominates(low, high, programs) for low in (Index(0), *self.lower)
                   for high in (Index(width), *self.upper))

    def subset_of(self, other, width, programs):
        # Sufficient symbolic implication, deliberately incomplete. Failure is not proof of a race.
        if self.empty(width, programs):
            return True
        return (all(any(dominates(a, b, programs) for a in (Index(0), *self.lower))
                    for b in other.lower)
                and all(any(dominates(b, a, programs) for a in (Index(width), *self.upper))
                        for b in other.upper))

    def full(self, width, programs):
        return Mask().subset_of(self, width, programs)


def comparison(predicate, lhs, rhs, programs):
    """Translate signed comparisons into lane >= lower / lane < upper constraints."""
    if not isinstance(lhs, Index) or not isinstance(rhs, Index):
        raise ValueError("data-dependent predicates are not in the rectangular-mask domain")
    if predicate in (6, 7, 8, 9):
        if lhs.bounds(programs)[0] < 0 or rhs.bounds(programs)[0] < 0:
            raise ValueError("unsigned comparison with a possibly negative operand is not proved")
        predicate = {6: 2, 7: 3, 8: 4, 9: 5}[predicate]
    delta = lhs.combine(rhs, -1)
    if delta.lane == 0:
        low, high = delta.bounds(programs)
        always = {0: low == high == 0, 1: high < 0 or low > 0, 2: high < 0,
                  3: high <= 0, 4: low > 0, 5: low >= 0}
        never = {0: high < 0 or low > 0, 1: low == high == 0, 2: low >= 0,
                 3: low > 0, 4: high <= 0, 5: high < 0}
        if always.get(predicate):
            return Mask()
        if never.get(predicate):
            return Mask.false()
        raise ValueError("program-dependent whole-tile predicate needs conditional-region support")
    if delta.lane == -1:
        delta = delta.scale(-1)
        predicate = {0: 0, 1: 1, 2: 4, 3: 5, 4: 2, 5: 3}[predicate]
    if delta.lane != 1:
        raise ValueError("predicate lane coefficient must be one after index normalization")
    bound = Index(-delta.constant, -delta.program)
    if predicate == 0:
        return Mask((bound,), (bound.combine(Index(1)),))
    if predicate == 2:
        return Mask(upper=(bound,))
    if predicate == 3:
        return Mask(upper=(bound.combine(Index(1)),))
    if predicate == 4:
        return Mask(lower=(bound.combine(Index(1)),))
    if predicate == 5:
        return Mask(lower=(bound,))
    raise ValueError("nonrectangular/complement predicate is not supported")
