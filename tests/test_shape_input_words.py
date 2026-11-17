"""Size programs consume exact declared runtime words under an explicit profile."""

from copy import deepcopy

import numpy as np
import pytest

from t2m.runtime_shapes import resolve_shapes
from t2m.shape_expressions import Program


def example():
    program = Program()
    word = program.intern("word", type_=64, word="w0")
    root = program.intern("index_cast", type_="index", args=[word])
    contract = {
        "schema": "t2m.runtime-shapes.v4",
        "inputs": [{"shape": [], "dtype": "i64"}],
        "outputs": [{"shape": [root], "dtype": "i32"}],
        "dimension_bindings": {},
        "word_bindings": {"w0": {"argument": 0, "indices": [], "dtype": "i64"}},
        "shape_program": program.export([root]),
    }
    return program, word, root, contract


@pytest.mark.parametrize("n", (0, 1, 7, 31))
def test_runtime_words_change_allocation_without_changing_program(n):
    _program, _word, _root, contract = example()
    original = deepcopy(contract)
    assert resolve_shapes(contract, [np.asarray(n, np.int64)]) == ((n,),)
    assert (
        contract == original
        and contract["shape_program"]["schema"] == "logical-size-program.v2"
    )


@pytest.mark.parametrize(
    "damage",
    (
        "profile",
        "program-schema",
        "old-program",
        "binding",
        "type",
        "index",
        "storage",
        "float",
        "negative",
        "word-type",
    ),
)
def test_words_cannot_be_claimed_under_older_profiles_or_unrelated_storage(damage):
    _program, word, _root, contract = example()
    values = [np.asarray(7, np.int64)]
    if damage == "profile":
        contract["schema"] = "t2m.runtime-shapes.v3"
    elif damage == "program-schema":
        contract["shape_program"]["schema"] = "logical-size-program.v1"
    elif damage == "old-program":
        contract.pop("word_bindings")
    elif damage == "binding":
        contract["word_bindings"]["w0"]["argument"] = 1
    elif damage == "type":
        contract["word_bindings"]["w0"]["dtype"] = "i32"
    elif damage == "index":
        contract["word_bindings"]["w0"]["indices"] = [0]
    elif damage == "storage":
        values = [np.asarray(7, np.int32)]
    elif damage == "float":
        values = [np.asarray(7, np.float64)]
    elif damage == "negative":
        values = [np.asarray(-1, np.int64)]
    else:
        contract["shape_program"]["nodes"][word]["type"] = 32
    with pytest.raises((ValueError, KeyError)):
        resolve_shapes(contract, values)


def test_word_solver_preserves_signed_bits_and_requires_actual_range_proof():
    pytest.importorskip("z3")
    program, word, root, _contract = example()
    zero = program.dimension(0)
    assert program.equivalent(root, zero, lambda _: [(0, 0)])
    assert not program.equivalent(root, zero, lambda _: [(0, 31)])
    assert not program.equivalent(root, root, lambda _: [(1 << 63, (1 << 64) - 1)])
    declaration = program.export([root])
    with pytest.raises(ValueError):
        Program.load(declaration, {}, [root])
    Program.load(declaration, {}, [root], words={"w0": 64})
    assert program.evaluate(word, {"w0": (1 << 64) - 1}) == (1 << 64) - 1
