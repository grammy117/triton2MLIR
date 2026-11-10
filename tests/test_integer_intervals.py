"""Division admission bounds every value, including poison domains and i64."""

from fractions import Fraction
import math

import pytest

from t2m.diagnostics import ConversionError
from t2m.integer_intervals import division_bounds


@pytest.mark.parametrize(
    "opcode", ["arith.divsi", "arith.ceildivsi", "arith.floordivsi"]
)
def test_every_signed_four_bit_interval_matches_complete_value_enumeration(opcode):
    function = {
        "arith.divsi": math.trunc,
        "arith.ceildivsi": math.ceil,
        "arith.floordivsi": math.floor,
    }[opcode]
    for lower in range(-8, 8):
        for upper in range(lower, 8):
            for dl in range(-8, 8):
                for du in range(dl, 8):
                    if dl <= 0 <= du or lower == -8 and dl <= -1 <= du:
                        with pytest.raises(ConversionError):
                            division_bounds(
                                (lower, upper), (dl, du), width=4, opcode=opcode
                            )
                    else:
                        values = [
                            function(Fraction(a, b))
                            for a in range(lower, upper + 1)
                            for b in range(dl, du + 1)
                        ]
                        assert division_bounds(
                            (lower, upper), (dl, du), width=4, opcode=opcode
                        ) == (min(values), max(values))


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_source_width_overflow_and_zero_are_not_promoted_away(width):
    minimum, maximum = -(1 << (width - 1)), (1 << (width - 1)) - 1
    with pytest.raises(ConversionError, match="overflow"):
        division_bounds((minimum, minimum), (-1, -1), width=width)
    with pytest.raises(ConversionError, match="zero divisor"):
        division_bounds((0, 1), (-1, 1), width=width)
    with pytest.raises(ConversionError, match="signed width"):
        division_bounds((0, maximum + 1), (1, 1), width=width)
    assert division_bounds((minimum, maximum), (1, 1), width=width) == (
        minimum,
        maximum,
    )
    assert division_bounds((maximum, maximum), (3, 3), width=width) == (
        maximum // 3,
        maximum // 3,
    )


def test_runtime_rounded_quotient_is_an_interval_not_a_linear_layout_fact():
    from t2m.shared_tensor import parse_shared
    from t2m.shared_indexing import Linear, linear, scalar_bounds

    module = parse_shared("""module {func.func @bound(%k:i32)->i32 {
      %c31=arith.constant 31:i32
      %c32=arith.constant 32:i32
      %n=arith.addi %k,%c31:i32
      %q=arith.divsi %n,%c32:i32
      return %q:i32
    }}""")
    function = next(op for op in module.walk() if op.name == "func.func")
    result = function.body.block.last_op.operands[0]
    facts = {function.body.block.args[0]: Linear(parameters=(("k", 0, 65, 1),))}
    with pytest.raises(ConversionError):
        linear(result, facts, (1,))
    assert scalar_bounds(result, facts, (1,)) == (0, 3)
    assert result not in facts


def test_affine_cancellation_cannot_hide_source_overflow_before_a_divide():
    from t2m.shared_tensor import parse_shared
    from t2m.shared_indexing import Linear, scalar_bounds

    module = parse_shared("""module {func.func @bound(%k:i32)->i32 {
      %two=arith.constant 2:i32
      %product=arith.muli %k,%two:i32
      %q=arith.divsi %product,%two:i32
      return %q:i32
    }}""")
    function = next(op for op in module.walk() if op.name == "func.func")
    facts = {
        function.body.block.args[0]: Linear(parameters=(("k", 1 << 30, 1 << 30, 1),))
    }
    with pytest.raises(ConversionError, match="overflow"):
        scalar_bounds(function.body.block.last_op.operands[0], facts, (1,))
