"""Executable source guards, dominance and unsigned words establish casts."""

import operator

import pytest
from xdsl.dialects import func

from t2m.shared_tensor import parse_shared
from t2m.source_index_ranges import _allowed, unsigned_index_range


def source(*, bridge="arith.index_castui", round_trip=True, bounds=True, other_root=False,
           after=False, nested=False, conjunction=False, upper=31):
    equality = "cf.assert %equal, \"arbitrary message\"" if round_trip else ""
    checks = "cf.assert %lower, \"lower\"\n cf.assert %upper, \"upper\"" if bounds else ""
    if conjunction and bounds:
        checks = '%both=arith.andi %lower,%upper:i1\n cf.assert %both,"both"'
    if nested:
        checks = "scf.if %flag {" + checks + "}"
    query = "%query=tensor.dim %x,%zero:tensor<?xi32>"
    # The private call marks an arbitrary binding point; later assertions must
    # not justify an adapter inserted there, even though return is later still.
    point = "%out=func.call @point(%query):(index)->index"
    body = (point + "\n" + checks) if after else (checks + "\n" + point)
    root = "%y" if other_root else "%x"
    return f'''module {{
      func.func @forward(%x:tensor<?xi32>,%y:tensor<?xi32>,%flag:i1)->index {{
        %zero=arith.constant 0:index
        %size=tensor.dim {root},%zero:tensor<?xi32>
        %word={bridge} %size:index to i64
        %restored={bridge} %word:i64 to index
        %equal=arith.cmpi eq,%size,%restored:index
        {equality}
        %two=arith.constant 2:i4
        %low=arith.extsi %two:i4 to i64
        %high=arith.constant {upper}:i64
        %lower=arith.cmpi sge,%word,%low:i64
        %upper=arith.cmpi ule,%word,%high:i64
        {query}
        {body}
        return %out:index
      }}
      func.func private @point(index)->index
    }}'''


def observation(text):
    module = parse_shared(text)
    module.verify()
    call = next(op for op in module.walk() if isinstance(op, func.CallOp))
    return unsigned_index_range(call.operands[0], call)


def test_duplicate_dimension_queries_retain_the_actual_guarded_range():
    assert observation(source()) == (2, 31)
    assert observation(source(conjunction=True)) == (2, 31)
    assert observation(source(upper=(1 << 40) - 1)) == (2, (1 << 40) - 1)


def test_enclosing_sequential_guards_dominate_nested_binding_points():
    nested = source().replace('%out=func.call @point(%query):(index)->index',
        '''%out=scf.execute_region -> index {
          %inner=func.call @point(%query):(index)->index
          scf.yield %inner:index
        }''')
    assert observation(nested) == (2, 31)


@pytest.mark.parametrize("options", [{"round_trip": False}, {"bridge": "arith.index_cast"},
                                    {"other_root": True}, {"nested": True}, {"after": True}])
def test_unchecked_truncated_signed_unrelated_or_nondominating_values_grant_no_narrow_range(options):
    result = observation(source(**options))
    assert result is None or result[1] >= 1 << 63


def test_advisory_tags_do_not_supply_unsigned_or_shape_facts():
    module = parse_shared(source(round_trip=False, bounds=False))
    from xdsl.dialects.builtin import StringAttr

    function = next(op for op in module.ops if isinstance(op, func.FuncOp))
    function.attributes["prov.shape_domain"] = StringAttr('{"min":2,"max":31}')
    call = next(op for op in module.walk() if isinstance(op, func.CallOp))
    call.attributes["prov.range"] = StringAttr("2..31")
    assert unsigned_index_range(call.operands[0], call) is None


@pytest.mark.parametrize("width", [2, 3, 4, 5])
def test_word_constraints_match_exhaustive_signed_and_unsigned_comparisons(width):
    relations = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
                 "le": operator.le, "gt": operator.gt, "ge": operator.ge}
    modulus, half = 1 << width, 1 << (width - 1)
    for predicate in ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"):
        signed = predicate.startswith("s")
        relation = relations[predicate[1:] if predicate[0] in "su" else predicate]
        for bound in range(modulus):
            allowed = _allowed(predicate, bound, width)
            for word in range(modulus):
                left = word - modulus if signed and word >= half else word
                right = bound - modulus if signed and bound >= half else bound
                assert any(low <= word <= high for low, high in allowed) == relation(left, right)
