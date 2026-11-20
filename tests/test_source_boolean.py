"""Independent truth tables check actual-SSA Boolean coverage proofs."""

import random
from itertools import product

from xdsl.dialects import arith, builtin, func
from xdsl.ir import Block

from t2m.source_boolean import BooleanCoverage


def test_small_boolean_dags_match_independent_truth_tables():
    block = Block(arg_types=[builtin.i1] * 3)
    assignments = list(product((False, True), repeat=3))
    mask = (1 << len(assignments)) - 1
    entries = [(value, sum(int(row[i]) << j for j, row in enumerate(assignments))) for i, value in enumerate(block.args)]
    for value in (0, 1):
        op = arith.ConstantOp.from_int_and_width(value, builtin.i1)
        entries.append((op.result, mask if value else 0))
    randomizer = random.Random(97)
    for _index in range(36):
        a, b, c = (randomizer.choice(entries) for _ in range(3))
        kind = randomizer.randrange(4)
        if kind == 3:
            operation, expected = arith.SelectOp(a[0], b[0], c[0]), (a[1] & b[1]) | ((mask ^ a[1]) & c[1])
        else:
            operation = (arith.AndIOp, arith.OrIOp, arith.XOrIOp)[kind](a[0], b[0])
            expected = (a[1] & b[1], a[1] | b[1], a[1] ^ b[1])[kind]
        entries.append((operation.results[0], expected))
    proof = BooleanCoverage()
    for left, right in product(entries, repeat=2):
        actual = proof.covers([proof.condition(right[0])], premise=proof.condition(left[0]))
        assert actual == (((mask ^ left[1]) | right[1]) == mask)


def test_integer_comparison_complements_keep_signedness_and_operand_direction():
    block = Block(arg_types=[builtin.i64, builtin.i64])
    x, y = block.args
    for left, right, swapped in (("eq", "ne", False), ("slt", "sge", False), ("ult", "uge", False),
                                  ("slt", "sle", True), ("ugt", "ule", False)):
        proof = BooleanCoverage()
        a = arith.CmpiOp(x, y, left)
        b = arith.CmpiOp(y if swapped else x, x if swapped else y, right)
        assert proof.covers([proof.condition(a.result), proof.condition(b.result)])
    proof = BooleanCoverage()
    assert not proof.covers([proof.condition(arith.CmpiOp(x, y, "slt").result),
                             proof.condition(arith.CmpiOp(x, y, "uge").result)])


def test_distinct_observations_and_provenance_cannot_supply_complementarity():
    proof = BooleanCoverage()
    first, second = (func.CallOp("observation", [], [builtin.i1]) for _ in range(2))
    for op in (first, second):
        op.attributes["prov.same_mask"] = builtin.StringAttr("claimed")
    a, b = proof.condition(first.results[0]), proof.condition(second.results[0])
    assert not proof.covers([a, proof.negate(b)])
    assert proof.covers([a, proof.negate(a)])


def test_unknown_guard_budget_cannot_supply_a_proof_or_disable_unconditional_coverage():
    proof = BooleanCoverage()
    block = Block(arg_types=[builtin.i1] * 130)
    values = [proof.condition(value) for value in block.args]
    assert values[-2:] == [None, None]
    assert not proof.covers([values[-1], proof.negate(values[-2])])
    assert proof.covers([1], premise=None)
