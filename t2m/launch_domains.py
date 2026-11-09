"""Affine bounds derived from the same runtime domains as the emitted forall.

MLIR's SCF ValueBounds model uses the actual induction lower/upper bounds.
Here p < ceil(D/q) gives q*p <= D-1 on every nonempty logical domain.
No physical launch, example extent or provenance assertion supplies this fact.
"""

from dataclasses import dataclass

from .dynamic_abi import GridExtent
from .shared_indexing import Linear


@dataclass(frozen=True)
class LaunchDomain:
    dimension: Linear
    divisor: int

    def __post_init__(self):
        if (not isinstance(self.dimension, Linear) or self.dimension.programs != (0, 0, 0)
                or self.dimension.loops or self.dimension.bounds(())[0] < 0
                or type(self.divisor) is not int or not 0 < self.divisor < 2**63):
            raise ValueError("launch domains require a nonnegative parameter dimension and positive divisor")


def launch_domains(abi):
    return tuple(LaunchDomain(abi.dimension(g.dimension), g.divisor) if isinstance(g, GridExtent) else None
                 for g in abi.grid)


def affine_bounds(form, grid, domains=()):
    """Intersect independent intervals with correlated logical launch bounds.

    Positive multiples of q admit an upper bound; negative multiples admit a
    lower bound. Other terms retain their existing conservative intervals.
    Empty launches have no memory observations. The production ABI guarantees
    that grid contains maxima for the very same checked dimensions.
    """
    original = form.bounds(grid)
    if not domains:
        return original
    bounds = []
    for maximize in (False, True):
        rewritten = form
        for axis, coefficient in enumerate(form.programs):
            domain = domains[axis] if axis < len(domains) else None
            if (not coefficient or domain is None or (coefficient > 0) != maximize
                    or coefficient % domain.divisor):
                continue
            term = Linear(programs=tuple(coefficient if i == axis else 0 for i in range(3)))
            rewritten = rewritten + term.scale(-1) + (domain.dimension + Linear(-1)).scale(coefficient // domain.divisor)
        bounds.append(rewritten.bounds(grid)[int(maximize)])
    return max(original[0], bounds[0]), min(original[1], bounds[1])
