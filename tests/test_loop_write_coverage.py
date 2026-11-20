"""Independent small-domain oracle for exact rectangular coverage proofs."""

from itertools import permutations, product
from math import prod

from t2m.loop_write_coverage import rectangular_bijection
from t2m.symbolic_layout import Polynomial


def domains(counts):
    return [(Polynomial.constant(0) if size == 1 else Polynomial.make([((("l", i, 0, size - 1),), 1)]),
             Polynomial.constant(size)) for i, size in enumerate(counts)]


def test_admitted_small_integer_maps_cover_every_ordinal_exactly_once():
    for counts in product(range(1, 4), repeat=2):
        loops = domains(counts)
        for coefficients in product(range(-1, 11), repeat=2):
            index = Polynomial.constant(0)
            for (induction, _upper), coefficient in zip(loops, coefficients, strict=True):
                index += induction.scale(coefficient)
            if rectangular_bijection(loops, index, grid=(1,)) is None:
                continue
            observed = [sum(i * c for i, c in zip(point, coefficients, strict=True))
                        for point in product(*(range(size) for size in counts))]
            assert sorted(observed) == list(range(prod(counts)))


def test_every_small_three_axis_mixed_radix_traversal_and_unit_axis_is_admitted():
    for counts in product(range(1, 4), repeat=3):
        loops = domains(counts)
        for order in permutations(range(3)):
            index, stride = Polynomial.constant(0), 1
            for axis in order:
                index += loops[axis][0].scale(stride)
                stride *= counts[axis]
            assert rectangular_bijection(loops, index, grid=(1,)) is not None


def test_unit_domain_facts_apply_only_when_the_original_store_is_reachable():
    size = Polynomial.make([((("v", "n", 0, 1),), 1)])
    count = Polynomial.make([((("v", "m", 0, 3),), 1)])
    induction = Polynomial.make([((("l", 0, 0, 2),), 1)])
    loops = [(induction, count), (Polynomial.constant(0), size)]
    assert rectangular_bijection(loops, induction * size, grid=(1,)) is not None
    assert rectangular_bijection(loops, induction * size + Polynomial.constant(1), grid=(1,)) is None


def test_negative_bounds_and_nonrectangular_cross_products_give_no_coverage():
    loops = domains((2, 3))
    assert rectangular_bijection(loops, loops[0][0] * loops[1][0], grid=(1,)) is None
    assert rectangular_bijection([(loops[0][0], Polynomial.constant(-2)), loops[1]], loops[0][0], grid=(1,)) is None
