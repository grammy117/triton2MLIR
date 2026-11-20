"""Checked upper intervals bound reachable IVs without specializing the loop."""

import pytest

from t2m.diagnostics import ConversionError
from t2m.loop_domains import bounded_loop, enclosed_loop, exit_induction_bounds, require_signed_loop


@pytest.mark.parametrize("lower,upper,step", [
    (0, (-5, 5), 1), (2, (-3, 8), 2), (-3, (-5, 8), 3), (1, (7, 7), 4),
])
def test_domain_contains_every_reachable_iv_and_keeps_zero_trip_count(lower, upper, step):
    domain = bounded_loop([(lower, lower), upper, (step, step)], width=32)
    counts = []
    for ub in range(upper[0], upper[1] + 1):
        values = range(lower, ub, step)
        counts.append(len(values))
        assert all(domain.induction_bounds[0] <= iv <= domain.induction_bounds[1] for iv in values)
    assert min(counts) == domain.minimum_iterations
    assert max(counts) == domain.maximum_iterations
    assert domain.exact == (upper[0] == upper[1])


@pytest.mark.parametrize("limits,diagnostic", [
    ([(0, 1), (0, 5), (1, 1)], "constant lower"),
    ([(0, 0), (0, 5), (1, 2)], "positive step"),
    ([(0, 0), (0, 5), (0, 0)], "positive step"),
    ([(0, 0), (0, 5), (-1, -1)], "positive step"),
    ([(0, 0), (-2, 0), (1, 1)], "no proved reachable iteration"),
    ([(0, 0), (0, 2**31 - 1), (2, 2)], "induction update may overflow"),
])
def test_unknown_or_overflowing_loop_premises_refuse(limits, diagnostic):
    with pytest.raises(ConversionError, match=diagnostic):
        bounded_loop(limits, width=32)


@pytest.mark.parametrize("lower,upper,step", [(0, (-7, 0), 1), (7, (-1, 7), 3), (-2, (-5, -2), 2)])
def test_empty_domains_return_no_induction_fact_and_preserve_the_initial_edge(lower, upper, step):
    domain = bounded_loop([(lower, lower), upper, (step, step)], width=32, allow_empty=True)
    assert domain.minimum_iterations == domain.maximum_iterations == 0
    assert all(not range(lower, end, step) for end in range(upper[0], upper[1] + 1))
    with pytest.raises(ConversionError, match="no reachable induction value"):
        _ = domain.induction_bounds


@pytest.mark.parametrize("field", ["properties", "attributes"])
def test_unsigned_comparison_cannot_reuse_a_signed_domain(field):
    from types import SimpleNamespace
    operation = SimpleNamespace(properties={}, attributes={})
    getattr(operation, field)["unsignedCmp"] = object()
    with pytest.raises(ConversionError, match="unsigned memory-loop comparison"):
        require_signed_loop(operation)


@pytest.mark.parametrize("lower,upper,step", [
    ((0, 3), (0, 8), (1, 3)), ((-5, 1), (-3, 7), (2, 4)),
    ((2, 4), (6, 9), (1, 1)), ((0, 0), (0, 8), (1, 3)),
    ((7, 9), (-3, 7), (1, 3)), ((-5, -3), (-7, 2), (1, 2)),
    ((0, 0), (8, 8), (125, 127)),
])
def test_interval_header_encloses_every_actual_iv_count_and_final_update(lower, upper, step):
    from itertools import product

    domain = enclosed_loop((lower, upper, step), width=8, allow_empty=True)
    assert domain.variable_header and not domain.exact
    counts = []
    for lb, ub, stride in product(*(range(lo, hi + 1) for lo, hi in (lower, upper, step))):
        values = range(lb, ub, stride)
        counts.append(len(values))
        assert domain.minimum_iterations <= len(values) <= domain.maximum_iterations
        assert -128 <= lb + len(values) * stride <= 127
        assert exit_induction_bounds(domain)[0] <= lb + len(values) * stride <= exit_induction_bounds(domain)[1]
        if values:
            low, high = domain.induction_bounds
            assert all(low <= value <= high for value in values)
    assert min(counts) == domain.minimum_iterations and max(counts) == domain.maximum_iterations
    if not domain.maximum_iterations:
        with pytest.raises(ConversionError, match="no reachable induction value"):
            _ = domain.induction_bounds


@pytest.mark.parametrize("limits,diagnostic", [
    (((0, 2), (0, 8), (0, 3)), "positive runtime step"),
    (((0, 2), (0, 8), (-2, -1)), "positive runtime step"),
    (((120, 121), (125, 126), (1, 5)), "induction update may overflow"),
    (((0, 1), (0, 128), (1, 3)), "signed integer width"),
    (((3, 2), (0, 8), (1, 3)), "signed integer width"),
])
def test_interval_header_rejects_nonpositive_steps_bad_bounds_and_final_overflow(limits, diagnostic):
    with pytest.raises(ConversionError, match=diagnostic):
        enclosed_loop(limits, width=8, allow_empty=True)


def test_constant_headers_retain_the_existing_tighter_stride_domain():
    limits = ((0, 0), (1, 127), (1, 1))
    assert enclosed_loop(limits, width=8) == bounded_loop(limits, width=8)
