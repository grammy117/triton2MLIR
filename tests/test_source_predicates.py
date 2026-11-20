"""Word partitions and boolean implications have independent value oracles."""

import operator

import pytest
from xdsl.dialects import scf

from t2m.shared_tensor import parse_shared
from t2m.source_predicates import branch_constraints, signed_comparison_enclosure


@pytest.mark.parametrize("width", (2, 3, 4))
def test_signed_enclosures_match_every_small_word_comparison(width):
    modulus, half = 1 << width, 1 << (width - 1)
    relations = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
                 "le": operator.le, "gt": operator.gt, "ge": operator.ge}
    for predicate in ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"):
        relation = relations[predicate[1:] if predicate[0] in "su" else predicate]
        for word in range(modulus):
            bound = word - modulus if predicate.startswith("s") and word >= half else word
            for low in range(-half, half):
                for high in range(low, half):
                    selected = [value for value in range(low, high + 1)
                        if relation(value if predicate.startswith("s") else value % modulus, bound)]
                    expected = (min(selected), max(selected)) if selected else None
                    assert signed_comparison_enclosure((low, high), predicate, word, width) == expected


@pytest.mark.parametrize("word", (0, 1, (1 << 31) - 1, 1 << 31, (1 << 32) - 1))
def test_disconnected_unsigned_and_nonzero_sets_keep_conservative_signed_hulls(word):
    full = (-(1 << 31), (1 << 31) - 1)
    result = signed_comparison_enclosure(full, "ne", word, 32)
    if word not in (1 << 31, (1 << 31) - 1):
        assert result == full
    assert signed_comparison_enclosure((0, 9), "ne", 0, 32) == (1, 9)
    assert signed_comparison_enclosure((-9, 0), "ne", 0, 32) == (-9, -1)
    assert signed_comparison_enclosure((-9, 9), "ne", 0, 32) == (-9, 9)


@pytest.mark.parametrize("opcode,truth,expected", (("andi", True, 2), ("andi", False, 0),
    ("ori", False, 2), ("ori", True, 0), ("xori", True, 1), ("xori", False, 1)))
def test_boolean_constraints_follow_necessary_implications_only(opcode, truth, expected):
    rhs = "%true" if opcode == "xori" else "%b"
    module = parse_shared(f'''module {{func.func @source(%x:i32,%y:i32) {{
      %zero=arith.constant 0:i32
      %true=arith.constant true
      %a=arith.cmpi sgt,%x,%zero:i32
      %b=arith.cmpi slt,%zero,%y:i32
      %condition=arith.{opcode} %a,{rhs}:i1
      scf.if %condition {{}} else {{}}
      func.return
    }}}}''')
    control = next(op for op in module.walk() if isinstance(op, scf.IfOp))
    constraints = branch_constraints(control.operands[0], truth)
    assert len(constraints) == expected
    if opcode == "xori":
        assert constraints[0][1] == ("sle" if truth else "sgt")
    elif constraints:
        assert all(row[1] == ("sgt" if truth else "sle") for row in constraints)
