"""Real source compare/exchange through the functional Merlin interface."""

import os

import numpy as np
import pytest

from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def capture(root, width=32, scalar=False, order="relaxed", scope="gpu", effect="readwrite", profile="atomics"):
    from t2m.examples.atomic_kernels import indexed_cas
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    constants = {"B": 8, "SCALAR": scalar, "SEM": order, "SCOPE": scope}
    source.write_text(capture_ttir(indexed_cas, {**{n: f"*i{width}" for n in ("state", "comparison", "value", "old")},
        **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("indexed_cas", tuple(TensorBuffer(name, (16,), effect, dtype=f"i{width}") for name, effect in (
        ("state", effect), ("comparison", "read"), ("value", "read"), ("old", "write"))), grid=(16 if scalar else 2,))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer",
                          address_policy="source-words", effect_profile=profile)


def cases(width):
    dtype = getattr(np, f"int{width}")
    result = []
    for shift in (0, -11, 23):
        incoming = np.resize(np.asarray([-(1 << (width - 1)), (1 << (width - 1)) - 1, 0, -13], dtype), 16)
        comparison = incoming.copy()
        comparison[1::2] ^= 1
        desired = np.arange(16, dtype=dtype) + shift
        result.append(([incoming, comparison, desired], [np.where(incoming == comparison, desired, incoming), incoming.copy()]))
    return result


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_cas_keeps_exact_success_failure_and_old_values(tmp_path, width, scalar):
    from test_integer_products import run_native, replay_library
    result = capture(tmp_path, width, scalar)
    assert "memref.generic_atomic_rmw" in result.mlir_text and "no_inline" in result.mlir_text
    domains = cases(width)
    run_native(tmp_path, result, domains)
    if MERLIN:
        replay_library(tmp_path, result, domains)


def translate(root, kernel, signature, constants, abi):
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, {**signature, **{key: "constexpr" for key in constants}}, constants))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer",
                          address_policy="source-words", effect_profile="atomics")


def check(root, result, domains):
    from test_integer_products import run_native, replay_library
    run_native(root, result, domains)
    if MERLIN:
        replay_library(root, result, domains)


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("scalar,lane_select", [(False, False), (True, False), (False, True), (True, True)])
def test_cas_selected_buffers_keep_parallel_conditional_effects(tmp_path, width, scalar, lane_select):
    from t2m.examples.atomic_kernels import selected_cas
    names = ("a", "b", "comparison", "value", "old")
    abi = TensorABI("selected_cas", tuple(TensorBuffer(name, (16,), "readwrite" if name in ("a", "b") else
        "write" if name == "old" else "read", dtype=f"i{width}") for name in names),
        grid=(16 if scalar else 2,), runtime_scalars=(RuntimeScalar("choice", "i1", 5),))
    result = translate(tmp_path, selected_cas, {**{name: f"*i{width}" for name in names}, "choice": "i1"},
        {"B": 8, "SCALAR": scalar, "LANE": lane_select}, abi)
    dtype = getattr(np, f"int{width}")
    domains = []
    for choice in (False, True):
        a, b = np.arange(16, dtype=dtype) - 13, np.arange(16, dtype=dtype) + 50
        selected = np.arange(16) % 2 != 0 if lane_select else np.full(16, choice)
        original = np.where(selected, b, a)
        comparison = original.copy()
        comparison[::3] ^= 1
        value = -np.arange(16, dtype=dtype) - 17
        updated = np.where(original == comparison, value, original)
        domains.append(([a, b, comparison, value, np.asarray(choice, np.bool_)],
                        [np.where(selected, a, updated), np.where(selected, updated, b), original]))
    assert "scf.if" in result.mlir_text
    check(tmp_path, result, domains)


@pytest.mark.parametrize("width", [16, 32, 64])
def test_rank_two_cas_keeps_original_tile_value_order(tmp_path, width):
    from t2m.examples.atomic_kernels import matrix_cas
    names = ("state", "comparison", "value", "old")
    abi = TensorABI("matrix_cas", tuple(TensorBuffer(name, (4, 4), "readwrite" if name == "state" else
        "write" if name == "old" else "read", dtype=f"i{width}") for name in names), grid=(2,))
    result = translate(tmp_path, matrix_cas, {name: f"*i{width}" for name in names}, {}, abi)
    domains = [([a.reshape(4, 4) for a in arrays], [a.reshape(4, 4) for a in expected]) for arrays, expected in cases(width)]
    check(tmp_path, result, domains)


def region_capture(root, width=32, scalar=False, while_=False):
    from t2m.examples.atomic_kernels import region_cas
    abi = TensorABI("region_cas", (TensorBuffer("state", (4, "n"), "readwrite", dtype=f"i{width}"),
        TensorBuffer("out", ("n",), "write", dtype=f"i{width}")), grid=(GridExtent("n", 1 if scalar else 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 16)), RuntimeScalar("steps", "i32", 3, bounds=(0, 3)),
                         RuntimeScalar("enabled", "i1", 4), RuntimeScalar("shift", "i32", 5, bounds=(-1000, 1000))))
    return translate(root, region_cas, {"state": f"*i{width}", "out": f"*i{width}", "n": "i32", "steps": "i32", "enabled": "i1", "shift": "i32"},
                     {"B": 8, "SCALAR": scalar, "WHILE": while_}, abi)


def region_arguments(n, steps, enabled, width=32, shift=0):
    dtype = getattr(np, f"int{width}")
    incoming = (np.arange(4, dtype=dtype)[:, None] - 11) + 100 * (np.arange(n, dtype=dtype)[None, :] % 2)
    state, total = incoming.copy(), np.zeros(n, dtype)
    if enabled:
        for row in range(steps):
            previous = state[row].copy()
            state[row] = np.where(previous == row - 11, row + 1, previous)
            total += previous
        previous = state[steps].copy()
        state[steps] = np.where(previous == total, total + 7, previous)
        total += previous
    return [incoming, np.asarray(n, np.int32), np.asarray(steps, np.int32), np.asarray(enabled, np.bool_),
            np.asarray(shift, np.int32)], [state, total]


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("scalar,while_", [(False, False), (True, False), (False, True), (True, True)])
def test_cas_regions_keep_old_value_recurrence_zero_trips_and_final_pointer(tmp_path, width, scalar, while_):
    result = region_capture(tmp_path, width, scalar, while_)
    domains = [region_arguments(n, steps, enabled, width, shift) for n, steps, enabled, shift in (
        (0, 3, True, 0), (1 if scalar else 8, 0, True, 0), (13 if scalar else 16, 3, True, 0), (8, 3, False, 1000))]
    check(tmp_path, result, domains)


@pytest.mark.parametrize("shift,n", [(-1, 16), (1000, 16), (0, 13)])
def test_unmasked_cas_does_not_invent_an_inactive_lane_for_bad_addresses(tmp_path, shift, n):
    from t2m.native import execute_native
    result = region_capture(tmp_path)
    arrays, _ = region_arguments(n, 3, False, shift=shift)
    arrays[3] = np.asarray(True, np.bool_)
    with pytest.raises(ValueError, match="exit -6"):
        execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=result.spec.runtime_shapes())


@pytest.mark.parametrize("width", [16, 32, 64])
def test_colliding_cas_has_one_winner_and_all_original_old_values(tmp_path, width):
    from t2m.examples.atomic_kernels import cas_collision
    dtype = getattr(np, f"int{width}")
    abi = TensorABI("cas_collision", (TensorBuffer("state", (1,), "readwrite", dtype=f"i{width}"),
        TensorBuffer("old", (16,), "write", dtype=f"i{width}")), grid=(16,))
    result = translate(tmp_path, cas_collision, {"state": f"*i{width}", "old": f"*i{width}"}, {}, abi)
    expected = np.ones(16, dtype)
    expected[0] = 0
    check(tmp_path, result, [([np.zeros(1, dtype)], [np.ones(1, dtype), expected])])
    if MERLIN:
        with np.load(tmp_path / "invoke0/evaluation/outputs.npz", allow_pickle=False) as actual:
            assert np.count_nonzero(actual["out1"] == 0) == 1
            assert np.all(actual["out1"][actual["out1"] != 0] == actual["out0"][0])


@pytest.mark.parametrize("order", ["relaxed", "acquire", "release", "acq_rel"])
@pytest.mark.parametrize("scope", ["gpu", "cta", "sys"])
def test_all_source_cas_orders_and_scopes_retain_their_contract(tmp_path, order, scope):
    result = capture(tmp_path, order=order, scope=scope)
    info = result.trace["memory"]["atomic_payload"][0]
    assert info["source_order"] == order and info["source_scope"] == scope
    check(tmp_path, result, cases(32))


@pytest.mark.parametrize("change", ["ordinary", "readonly", "fresh"])
def test_cas_requires_explicit_effects_writable_state_and_completed_initialization(tmp_path, change):
    from t2m.diagnostics import ConversionError
    from t2m.native import execute_native
    if change == "ordinary":
        with pytest.raises(ConversionError, match="separately qualified effect"):
            capture(tmp_path, profile="ordinary")
    elif change == "readonly":
        with pytest.raises(ConversionError, match="read-only"):
            capture(tmp_path, effect="read")
    else:
        result = capture(tmp_path, effect="write")
        arrays, _ = cases(32)[0]
        with pytest.raises(ValueError, match="exit -6"):
            execute_native(result.mlir_text, arrays[1:], tmp_path / "native", python=NATIVE)


@pytest.mark.parametrize("attribute", ["sem", "scope"])
def test_rehashed_native_cas_event_cannot_change_the_source_contract(tmp_path, attribute):
    import json
    from xdsl.dialects import builtin
    from t2m.diagnostics import ConversionError
    from t2m.shared_tensor import opcode, parse_shared
    from t2m.word_events import print_ir
    from t2m.word_evidence import check_word_stage, record, replay_events
    result = capture(tmp_path)
    stage = json.loads(result.auxiliary_artifacts["word-stage.json"])
    events = parse_shared(stage["events"]["output"])
    atomic = next(op for op in events.walk() if opcode(op) == "tt.atomic_cas")
    atomic.attributes.pop(attribute, None)
    atomic.properties[attribute] = builtin.IntegerAttr(2, builtin.i32)
    stage["events"]["output"] = print_ir(events)
    stage["events"]["native_stage"]["artifacts"]["lowered.mlir"] = record(stage["events"]["output"])
    source, output = (result.auxiliary_artifacts[n] for n in ("word-input.mlir", "word-tensor.mlir"))
    check_word_stage(stage, source, output)
    with pytest.raises(ConversionError, match="source operation/order/scope"):
        replay_events(stage, source)


def test_visible_compare_exchange_demo_preserves_the_complete_tensor_package(tmp_path):
    import json
    from t2m.bundle import verify_bundle
    from t2m.cli import shared_demo
    out = tmp_path / "cas-demo"
    summary = shared_demo(out, kernel="atomic-cas", optimizer=OPT, python=NATIVE,
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile="atomics")
    assert summary["native_numerical_check"] == "passed"
    assert verify_bundle(out)["profile"] == "shared-source-word-buffer"
    differences = json.loads((out / "compatibility.json").read_text())["differences"]
    assert any("compare/exchange" in difference for difference in differences)
    assert "memref.generic_atomic_rmw" in (out / "model.mlir").read_text()
    assert '"llvm.' not in (out / "model.mlir").read_text()
