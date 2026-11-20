"""Original CFG and synchronous tile dependencies survive the same interface."""

import json
import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
TOOLS = all(os.environ.get(key) for key in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


def capture(root, dtype="i64", effect="ordinary", memory="guarded-buffer", allocation=False):
    from t2m.examples.cfg_barrier_kernels import early_tile_exchange
    from t2m.frontend import capture_ttir

    atomic, volatile = effect in ("typed-atomics", "atomic-volatile"), effect in ("volatile", "atomic-volatile")
    source = root / "source.ttir"
    source.write_text(capture_ttir(early_tile_exchange,
        {**{name: "*" + ("fp32" if dtype == "f32" else "i64") for name in ("state", "old", "seen")},
         "n": "i32", "steps": "i32", "mode": "i32", "B": "constexpr", "ATOMIC": "constexpr", "VOLATILE": "constexpr"},
        {"B": 4, "ATOMIC": atomic, "VOLATILE": volatile}))
    abi = TensorABI("allocation_early_tile_exchange" if allocation else "early_tile_exchange",
        tuple(TensorBuffer(name, ("n", 4), "readwrite", dtype)
        for name in ("state", "old", "seen")), grid=(GridExtent("n", 4),),
        runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 19)),
                         RuntimeScalar("steps", "i32", 4, bounds=(0, 4)),
                         RuntimeScalar("mode", "i32", 5, bounds=(0, 1))))
    options = dict(optimizer=OPT, memory_profile=memory,
        address_policy="source-words" if memory == "guarded-buffer" else "structured", effect_profile=effect,
        f32_atomic_add="ieee" if atomic and dtype == "f32" else None)
    if allocation:
        from t2m.allocation_bindings import PointerArgument, ScalarArgument, convert_allocations
        return convert_allocations(source, abi,
            (*[PointerArgument(name) for name in ("state", "old", "seen")],
             *[ScalarArgument(name) for name in ("n", "steps", "mode")]), root / "translation",
            source_entry="early_tile_exchange", **options)
    return convert_shared(source, abi, root / "translation", **options)


def arguments(dtype="i64", n=19, steps=3, mode=0, pattern=0):
    dt = np.int64 if dtype == "i64" else np.float32
    state = np.arange(n * 4, dtype=dt).reshape(n, 4) - pattern - 7
    if dtype == "i64":
        state += 1 << 55
    old, seen = np.full_like(state, -17), np.full_like(state, -23)
    expected = [value.copy() for value in (state, old, seen)]
    for origin in range(0, n, 4):
        if (origin // 4 + mode) % 2 == 0:
            continue
        stop = min(origin + 4, n)
        for step in range(steps):
            expected[1][origin:stop] = expected[0][origin:stop]
            expected[0][origin:stop] += step + 1
            tile = np.full((4, 4), -3, dtype=dt)
            tile[:stop - origin] = expected[0][origin:stop]
            expected[2][origin:stop] = tile.T[:stop - origin]
    return [state, old, seen, *[np.asarray(x, np.int32) for x in (n, steps, mode)]], expected


@pytest.mark.skipif(not (OPT and NATIVE), reason="existing source/native tools required")
@pytest.mark.parametrize("effect", ["ordinary", "typed-atomics", "volatile", "atomic-volatile"])
@pytest.mark.parametrize("dtype", ["i64", "f32"])
def test_native_early_exit_retains_all_program_phases_and_results(tmp_path, dtype, effect):
    from t2m.native import execute_native

    result = capture(tmp_path, dtype, effect)
    assert any(row["opcode"] == "cf.cond_br" for row in result.trace["source_operations"])
    assert "control_flow_normalization" in result.trace
    assert len([row for row in result.trace["source_operations"] if row["opcode"] == "ttg.barrier"]) == 2
    assert all(row["program_barrier"]["source_control"] == "whole-program-scalar-cfg-v1"
               for row in result.trace["source_operations"] if row["opcode"] == "ttg.barrier")
    assert "scf.forall" in result.mlir_text and "ttg.barrier" not in result.mlir_text
    for index, (n, steps, mode) in enumerate(((0, 3, 0), (1, 2, 1), (7, 0, 1), (13, 3, 0), (19, 4, 1))):
        inputs, expected = arguments(dtype, n, steps, mode, index)
        actual, _ = execute_native(result.mlir_text, inputs, tmp_path / f"native-{index}", python=NATIVE,
                                  shape_abi=result.spec.runtime_shapes())
        for a, b in zip(actual, expected, strict=True):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


@pytest.mark.skipif(not TOOLS, reason="existing source/native/Merlin tools required")
@pytest.mark.parametrize("dtype,effect", [("i64", "ordinary"), ("i64", "typed-atomics"),
    ("f32", "volatile"), ("f32", "atomic-volatile")])
def test_protected_cfg_phases_keep_independent_source_custody_and_merlin_execution(tmp_path, dtype, effect):
    from test_integer_products import replay_library

    result = capture(tmp_path, dtype, effect, allocation=True)
    protection = json.loads(result.auxiliary_artifacts["allocation-protection.json"])
    assert len(protection["barrier_conversions"]) == 2
    assert all(phase["contract"]["source_control"] == "whole-program-scalar-cfg-v1"
               for phase in protection["barrier_conversions"])
    cases = [arguments(dtype, n, steps, mode, index)
        for index, (n, steps, mode) in enumerate(((0, 3, 0), (1, 2, 1), (13, 3, 0), (19, 4, 1)))]
    replay_library(tmp_path, result, cases)
    observations = [(tmp_path / "merlin/host/outputs.npz", cases[-1][1]),
                    *[(tmp_path / f"invoke{i}/evaluation/outputs.npz", expected)
                      for i, (_, expected) in enumerate(cases)]]
    for path, expected in observations:
        with np.load(path, allow_pickle=False) as actual:
            assert len(actual.files) == 3
            for i, b in enumerate(expected):
                a = actual[f"out{i}"]
                assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


def cfg_function():
    from t2m.shared_tensor import opcode, parse_shared

    module = parse_shared('''"builtin.module"() ({
      "tt.func"() <{sym_name="f",function_type=(i1,i64)->()}> ({
      ^entry(%choose:i1,%x:i64):
        cf.cond_br %choose, ^body(%x:i64), ^done
      ^body(%carry:i64):
        "ttg.barrier"() <{addrSpace=31:i32}>:()->()
        cf.br ^done
      ^done:
        "tt.return"():()->()
      }):()->()
    }):()->()''')
    function = next(op for op in module.walk() if opcode(op) == "tt.func")
    return function


def test_scalar_cfg_participation_preserves_actual_edges_and_types():
    from t2m.program_barriers import cfg_participation

    assert cfg_participation(cfg_function()) == "whole-program-scalar-cfg-v1"


@pytest.mark.parametrize("damage", ["condition", "argument", "missing-argument", "foreign-block", "terminator"])
def test_cfg_participation_does_not_grant_divergent_or_malformed_edges(damage):
    from t2m.program_barriers import cfg_participation
    from xdsl.dialects import builtin
    from xdsl.ir import Block

    function = cfg_function()
    blocks = list(function.regions[0].blocks)
    branch = blocks[0].last_op
    if damage == "condition":
        branch.operands[0]._type = builtin.TensorType(builtin.i1, [1])
    elif damage == "argument":
        blocks[1].args[0]._type = builtin.i32
    elif damage == "missing-argument":
        blocks[1].insert_arg(builtin.i64, 1)
    elif damage == "foreign-block":
        branch.successors = (Block(arg_types=[builtin.i64]), branch.successors[1])
    else:
        terminal = blocks[1].last_op
        blocks[1].erase_op(terminal)
    with pytest.raises(ValueError, match="CFG"):
        cfg_participation(function)


@pytest.mark.skipif(not TOOLS, reason="existing source/native/Merlin tools required")
def test_cfg_participation_claims_must_match_actual_original_source_in_both_readers(tmp_path):
    from copy import deepcopy
    from test_atomic_barriers import merlin_refusal
    from t2m.source_protection import validate

    result = capture(tmp_path, allocation=True)
    artifacts = result.auxiliary_artifacts
    protection = json.loads(artifacts["allocation-protection.json"])
    outcomes = []
    for damage in ("missing-control", "different-control", "different-scope"):
        changed = deepcopy(protection)
        for phase in changed["barrier_conversions"]:
            if damage == "missing-control":
                phase["contract"].pop("source_control")
            elif damage == "different-control":
                phase["contract"]["source_control"] = "per-lane"
            else:
                phase["contract"]["participants"] = "all-logical-programs"
        by_id = {phase["contract"]["source_id"]: phase["contract"] for phase in changed["barrier_conversions"]}
        for row in changed["source_graph"]["operations"]:
            if row["id"] in by_id:
                row["program_barrier"] = deepcopy(by_id[row["id"]])
        members = {**artifacts, "allocation-protection.json": json.dumps(changed)}
        with pytest.raises(ValueError, match="barrier") as producer:
            validate(members, artifacts["allocation-original.ttir"], "early_tile_exchange")
        consumer = merlin_refusal(tmp_path, members, source_entry="early_tile_exchange")
        assert "barrier" in consumer
        outcomes.append({"damage": damage, "producer": str(producer.value), "consumer": consumer})
    (tmp_path / "control-refusals.json").write_text(json.dumps(outcomes, indent=2) + "\n")
