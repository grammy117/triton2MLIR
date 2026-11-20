"""An actual unit recurrence is an ordinal; count intervals alone are not."""

import json
import os

import numpy as np
import pytest
from test_value_exit_coordinates import projected_exit

from t2m.diagnostics import ConversionError


@pytest.mark.parametrize("width", [32, 64])
def test_zero_plus_one_result_keeps_the_original_loop_and_its_own_fact(width):
    module = projected_exit(width, initial="%zero", update="%ordinal")
    function = next(iter(module.ops))
    loop = next(op for op in module.walk() if op.name == "scf.for")
    assert tuple(loop.operands[:3]) == tuple(function.body.block.args[1:4])
    evidence = json.loads(function.attributes["prov.source_loops"].data)[0]
    assert evidence["actual_iteration_ordinal_results"] == [0]
    assert not evidence.get("actual_exit_induction_results")


@pytest.mark.parametrize("changes", [{"initial": "%lo"}, {"update": "%wrong_ordinal"},
    {"update": "%step_ordinal"}, {"slot": 1}, {"bounds": None}, {"bounds": (0, 3)}, {"unsigned": True}])
def test_unproved_initial_increment_tuple_slot_or_header_cannot_supply_an_ordinal(changes):
    options = {"initial": "%zero", "update": "%ordinal", **changes}
    with pytest.raises(ConversionError):
        projected_exit(**options)


def test_source_counter_width_must_cover_the_actual_maximum_count():
    # Header IVs and their final update fit i8. The 0..255 mathematical count
    # does not fit the carried signed i8 value, even though the root is large
    # enough for that false enclosure. Numeric wrap is not an ordinal proof.
    with pytest.raises(ConversionError):
        projected_exit(8, initial="%zero", update="%ordinal", lower=(-128, -120), upper=(120, 127),
            bounds=(1, 1), capacity=256)


def test_a_small_carrier_is_valid_when_its_count_fits_even_with_negative_header_ivs():
    module = projected_exit(8, initial="%zero", update="%ordinal", lower=(-128, -120), upper=(-16, -8),
        bounds=(1, 1), capacity=256)
    evidence = json.loads(next(iter(module.ops)).attributes["prov.source_loops"].data)[0]
    assert evidence["actual_iteration_ordinal_results"] == [0]
    module.verify()


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native environment required")
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("double", [False, True])
def test_native_count_is_actual_executions_not_exit_iv_or_maximum(tmp_path, width, double):
    from t2m.ir import text
    from t2m.native import execute_native

    module = projected_exit(width, initial="%zero", update="%ordinal", double=double, second_ordinal=double)
    x = np.arange(32, dtype=np.int64) * 7919 + (1 << 56)
    integer = np.int32 if width == 32 else np.int64
    for index, (lo, hi, stride) in enumerate(((1, 8, 3), (3, 1, 2), (0, 0, 1), (0, 8, 1))):
        arrays = [x, *[np.asarray(value, integer) for value in (lo, hi, stride, 2)]]
        (actual,), _ = execute_native(text(module), arrays, tmp_path / f"native-{index}", python=os.environ["T2M_NATIVE_PYTHON"])
        count = len(range(lo, hi, stride))
        if double:
            count += len(range(2, hi, stride))
        assert (actual.shape, actual.dtype, actual.tobytes()) == ((1,), x.dtype, x[count:count+1].tobytes())
