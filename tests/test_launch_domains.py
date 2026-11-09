"""Check affine launch inequalities against independent finite-domain oracles."""

from itertools import product
import random

from t2m.launch_domains import LaunchDomain, affine_bounds
from t2m.shared_indexing import Linear


def dimension(name, high):
    return Linear(parameters=((name, 0, high, 1),))


def test_correlated_bounds_enclose_actual_domains_with_signed_coefficients():
    n = dimension("n", 9)
    for divisor, coefficient, parameter, constant in product(range(1, 5), range(-5, 6), range(-3, 4), range(-5, 6)):
        grid = ((9 + divisor - 1) // divisor,)
        form = Linear(constant, (coefficient, 0, 0), parameters=(("n", 0, 9, parameter),))
        lower, upper = affine_bounds(form, grid, (LaunchDomain(n, divisor),))
        observed = [constant + coefficient * p + parameter * size
                    for size in range(10) for p in range((size + divisor - 1) // divisor)]
        assert lower <= min(observed) <= max(observed) <= upper
        assert affine_bounds(form, grid) == form.bounds(grid)


def test_multiple_correlated_axes_enclose_joint_actual_launches():
    rng = random.Random(581)
    n, m = dimension("n", 5), dimension("m", 4)
    for _ in range(96):
        q0, q1 = rng.randrange(1, 5), rng.randrange(1, 5)
        p0, p1, p2 = [rng.randrange(-5, 6) for _ in range(3)]
        bn, bm, constant = [rng.randrange(-4, 5) for _ in range(3)]
        form = Linear(constant, (p0, p1, p2), parameters=(("n", 0, 5, bn), ("m", 0, 4, bm)))
        grid = ((5 + q0 - 1) // q0, (4 + q1 - 1) // q1, 3)
        lower, upper = affine_bounds(form, grid, (LaunchDomain(n, q0), LaunchDomain(m, q1), None))
        observed = [constant + p0 * a + p1 * b + p2 * c + bn * nv + bm * mv
                    for nv in range(6) for mv in range(5)
                    for a in range((nv + q0 - 1) // q0) for b in range((mv + q1 - 1) // q1) for c in range(3)]
        assert lower <= min(observed) <= max(observed) <= upper


def test_relation_proves_unit_rows_but_not_shifted_or_unclipped_ragged_tiles():
    n = dimension("n", 9)
    unit = (LaunchDomain(n, 1),)
    assert affine_bounds(Linear(1, (1, 0, 0), parameters=(("n", 0, 9, -1),)), (9,), unit)[1] == 0
    assert affine_bounds(Linear(2, (1, 0, 0), parameters=(("n", 0, 9, -1),)), (9,), unit)[1] > 0
    ragged = (LaunchDomain(n, 4),)
    assert affine_bounds(Linear(4, (4, 0, 0), parameters=(("n", 0, 9, -1),)), (3,), ragged)[1] > 0
    reverse = Linear(-1, (-1, 0, 0), parameters=(("n", 0, 9, 1),))
    assert affine_bounds(reverse, (9,), unit)[0] == 0
    wrong_dimension = (LaunchDomain(dimension("m", 9), 1),)
    assert affine_bounds(Linear(1, (1, 0, 0), parameters=(("n", 0, 9, -1),)), (9,), wrong_dimension)[1] > 0
