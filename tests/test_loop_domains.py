"""Checked upper intervals bound reachable IVs without specializing the loop."""

import pytest

from t2m.diagnostics import ConversionError
from t2m.loop_domains import bounded_loop, require_signed_loop


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


@pytest.mark.parametrize("field", ["properties", "attributes"])
def test_unsigned_comparison_cannot_reuse_a_signed_domain(field):
    from types import SimpleNamespace
    operation = SimpleNamespace(properties={}, attributes={})
    getattr(operation, field)["unsignedCmp"] = object()
    with pytest.raises(ConversionError, match="unsigned memory-loop comparison"):
        require_signed_loop(operation)
