"""Executable word semantics, conservative ranges and explicit size programs."""

from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from t2m.runtime_shapes import resolve_shapes
from t2m.shape_expressions import BINARY, MAX_SIZE, PREDICATES, Program, signed


def word(program, axis, width):
    return program.intern("index_castui", type_=width, args=[program.dimension(axis)])


def oracle(op, a, b, width):
    modulus = 1 << width
    if op == "addi":
        return (a + b) % modulus
    if op == "subi":
        return (a - b) % modulus
    if op == "muli":
        return (a * b) % modulus
    if op == "andi":
        return a & b
    if op == "ori":
        return a | b
    if op == "xori":
        return a ^ b
    if op == "divui":
        return a // b
    if op == "remui":
        return a % b
    a, b = signed(a, width), signed(b, width)
    if op != "remsi" and a == -(1 << (width - 1)) and b == -1:
        raise ValueError("unqualified signed overflow")
    if op == "floordivsi":
        return (a // b) % modulus
    quotient = (-1 if (a < 0) != (b < 0) else 1) * (abs(a) // abs(b))
    return (quotient if op == "divsi" else a - quotient * b) % modulus


@pytest.mark.parametrize("op", sorted(BINARY))
@pytest.mark.parametrize("width", (1, 2, 4))
def test_small_words_match_independent_arithmetic_and_range_enclosures(op, width):
    program = Program()
    left, right = word(program, "a", width), word(program, "b", width)
    node = program.intern(op, type_=width, args=[left, right])
    ranges = [(0, (1 << width) - 1), (0, 0), ((1 << width) - 1, (1 << width) - 1)]
    if width > 1:
        ranges.extend(
            [(0, (1 << (width - 1)) - 1), (1 << (width - 1), (1 << width) - 1)]
        )
    for a_range, b_range in product(ranges, repeat=2):
        enclosure = program.bounds(
            node, lambda name: a_range if name == "a" else b_range
        )
        for a, b in product(
            range(a_range[0], a_range[1] + 1), range(b_range[0], b_range[1] + 1)
        ):
            try:
                expected = oracle(op, a, b, width)
            except (ValueError, ZeroDivisionError):
                with pytest.raises(ValueError):
                    program.evaluate(node, {"a": a, "b": b})
                continue
            assert program.evaluate(node, {"a": a, "b": b}) == expected
            assert enclosure is None or any(
                low <= expected <= high for low, high in enclosure
            )


@pytest.mark.parametrize("predicate", PREDICATES)
def test_signed_unsigned_predicates_and_conditional_range_refinement(predicate):
    program, width = Program(), 4
    value = word(program, "n", width)
    bound = program.intern("const", type_=width, value=11)
    condition = program.intern("cmpi", args=[value, bound], predicate=predicate)
    zero = program.intern("const", type_=width, value=0)
    selected = program.intern("select", type_=width, args=[condition, value, zero])
    enclosure = program.bounds(selected, lambda _: (0, 15))
    for n in range(16):
        a, b = (signed(n, width), -5) if predicate.startswith("s") else (n, 11)
        relation = predicate if predicate in ("eq", "ne") else predicate[1:]
        expected = {
            "eq": a == b,
            "ne": a != b,
            "lt": a < b,
            "le": a <= b,
            "gt": a > b,
            "ge": a >= b,
        }[relation]
        assert program.evaluate(condition, {"n": n}) == int(expected)
        result = n if expected else 0
        assert program.evaluate(selected, {"n": n}) == result
        assert any(low <= result <= high for low, high in enclosure)


@pytest.mark.parametrize("op,target", [("extui", 8), ("extsi", 8), ("trunci", 2)])
def test_casts_keep_original_sign_and_word_width(op, target):
    program = Program()
    source = word(program, "n", 4)
    cast = program.intern(op, type_=target, args=[source])
    intervals = program.bounds(cast, lambda _: (0, 15))
    for n in range(16):
        expected = (signed(n, 4) if op == "extsi" else n) % (1 << target)
        assert program.evaluate(cast, {"n": n}) == expected
        assert any(low <= expected <= high for low, high in intervals)


def example():
    program = Program()
    source = word(program, "d0_0", 64)
    divisor = program.intern("const", type_=64, value=2)
    quotient = program.intern("divui", type_=64, args=[source, divisor])
    root = program.intern("index_castui", type_="index", args=[quotient])
    contract = {
        "schema": "t2m.runtime-shapes.v3",
        "inputs": [{"shape": ["d0_0"], "dtype": "i32"}],
        "outputs": [{"shape": [root], "dtype": "i32"}],
        "dimension_bindings": {"d0_0": {"argument": 0, "axis": 0}},
        "shape_program": program.export([root]),
    }
    return program, root, contract


def test_runtime_evaluation_uses_only_bound_axes_and_retains_dag_sharing():
    program, root, contract = example()
    for n in (0, 1, 7, 31):
        assert resolve_shapes(contract, [np.zeros(n, np.int32)]) == ((n // 2,),)
    doubled = word(program, "d0_0", 64)
    for _ in range(30):
        doubled = program.intern("addi", type_=64, args=[doubled, doubled])
    assert program.evaluate(doubled, {"d0_0": 1}) == 1 << 30
    assert program.visits < 100


@pytest.mark.parametrize(
    "damage",
    (
        "hash",
        "unknown",
        "width",
        "axis",
        "cycle",
        "unaccounted",
        "root-type",
        "fields",
        "schema",
    ),
)
def test_program_reader_refuses_incomplete_or_rehashed_semantic_declarations(damage):
    program, root, contract = example()
    declaration = deepcopy(contract["shape_program"])
    if damage == "hash":
        declaration["nodes"][root]["op"] = "index_cast"
    elif damage == "unknown":
        declaration["nodes"][root]["args"] = ["missing"]
    elif damage == "width":
        declaration["nodes"][root]["type"] = 4097
    elif damage == "axis":
        next(row for row in declaration["nodes"].values() if row["op"] == "axis")[
            "axis"
        ] = "sample13"
    elif damage == "cycle":
        declaration["nodes"][root]["args"] = [root]
    elif damage == "unaccounted":
        declaration["nodes"]["unused"] = {"op": "const", "type": "index", "value": 13}
    elif damage == "root-type":
        root = program.nodes[root]["args"][0]
    elif damage == "fields":
        declaration["nodes"][root]["permission"] = "trusted"
    else:
        declaration["schema"] = "logical-size-program.v0"
    with pytest.raises(ValueError):
        Program.load(declaration, contract["dimension_bindings"], [root])


def test_select_evaluates_only_the_taken_branch_and_limits_recursive_programs():
    program = Program()
    zero, one = (program.intern("const", type_=8, value=n) for n in (0, 1))
    undefined = program.intern("divui", type_=8, args=[one, zero])
    condition = program.intern("const", type_=1, value=1)
    selected = program.intern("select", type_=8, args=[condition, one, undefined])
    assert program.evaluate(selected, {}) == 1
    deep = word(program, "n", 64)
    mask = program.intern("const", type_=64, value=7)
    for _ in range(65):
        deep = program.intern("xori", type_=64, args=[deep, mask])
    with pytest.raises(ValueError, match="budget"):
        program.evaluate(deep, {"n": 1})


def test_source_solver_checks_word_wrap_and_actual_assertions_without_sampling():
    pytest.importorskip("z3")
    program = Program()
    a, b = program.dimension("a"), program.dimension("b")
    equal = program.intern("cmpi", predicate="eq", args=[a, b])
    assert not program.equivalent(a, b, lambda _: (0, MAX_SIZE))
    assert program.equivalent(a, b, lambda _: (0, MAX_SIZE), assertions=[equal])
    narrow = word(program, "a", 4)
    restored = program.intern("index_castui", type_="index", args=[narrow])
    assert program.equivalent(a, restored, lambda _: (0, 15))
    assert not program.equivalent(a, restored, lambda _: (0, 16))


def test_missing_or_inconclusive_solver_never_supplies_shape_authority(monkeypatch):
    import builtins

    z3 = pytest.importorskip("z3")
    program = Program()
    a, b = program.dimension("a"), program.dimension("b")
    solver = z3.Solver()
    monkeypatch.setattr(solver, "check", lambda: z3.unknown)
    monkeypatch.setattr(z3, "Solver", lambda: solver)
    with pytest.raises(ValueError, match="not proved"):
        program.equivalent(a, b, lambda _: (0, 31))
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "z3":
            raise ImportError("optional shape solver unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ValueError, match="extra"):
        program.equivalent(a, b, lambda _: (0, 31))
