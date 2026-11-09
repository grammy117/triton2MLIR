"""Floating exchange is a bitwise effect, including NaN payloads and zeros."""

from copy import deepcopy
import json
import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
NATIVE_REQUIRED = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def capture(root, width=32, scalar=False, *, operation="exch", effect_profile="typed-atomics", order="relaxed", scope="gpu"):
    from t2m.examples.atomic_kernels import indexed_atomic
    from t2m.frontend import capture_ttir
    dtype, pointer = f"f{width}", f"*fp{width}"
    constants = {"B": 8, "OP": operation, "SCALAR": scalar, "MASK": True, "FRESH": False, "SEM": order, "SCOPE": scope}
    source = root / "source.ttir"
    source.write_text(capture_ttir(indexed_atomic, {"x": pointer, "index": "*i32", "state": pointer, "old": pointer,
        "n": "i32", "m": "i32", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("indexed_atomic", (TensorBuffer("x", ("n",), dtype=dtype), TensorBuffer("index", ("n",), dtype="i32"),
        TensorBuffer("state", ("m",), "readwrite", dtype=dtype), TensorBuffer("old", ("n",), "write", dtype=dtype)),
        grid=(GridExtent("n", 1 if scalar else 8),), runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
        RuntimeScalar("m", "i32", 5, bounds=(0, 37))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile=effect_profile)


def domain(n, width):
    # Construct and compare words, so signaling NaNs never pass through host FP arithmetic.
    uint = getattr(np, f"uint{width}")
    fp = getattr(np, f"float{width}")
    if width == 32:
        patterns = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 0xffc05678, 0x7f801234,
                    0xff805678, 1, 0x80000001, 0x7f7fffff, 0xff7fffff]
    else:
        patterns = [0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
                    0x7ff8000000001234, 0xfff8000000005678, 0x7ff0000000001234, 0xfff0000000005678,
                    1, 0x8000000000000001, 0x7fefffffffffffff, 0xffefffffffffffff]
    # Rotate across domains so a representation excluded by one mask is active
    # in another; every special word must actually participate in an exchange.
    values = np.resize(np.roll(np.array(patterns, dtype=uint), n // 8), n)
    incoming = np.resize(np.roll(np.array(patterns[::-1], dtype=uint), n // 8), 37)
    index = np.arange(n, dtype=np.int32)
    enabled = index % 3 != 1
    index[~enabled] = -1000
    state, old = incoming.copy(), np.zeros(n, dtype=uint)
    for i in range(n):
        if enabled[i]:
            old[i], state[i] = state[index[i]], values[i]
    return [values.view(fp), index, incoming.view(fp), np.asarray(n, np.int32), np.asarray(37, np.int32)], [state, old]


@NATIVE_REQUIRED
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_native_exchange_preserves_all_bits_and_inactive_addresses(tmp_path, width, scalar):
    from t2m.native import execute_native
    result = capture(tmp_path, width, scalar)
    admission = json.loads(result.auxiliary_artifacts["memory-admission.json"])
    assert admission["schema"] == "t2m.memory-admission.v4"
    info = result.trace["memory"]["atomic_payload"][0]
    assert info["numeric_contract"] == "bitwise-exchange" and info["value_dtype"] == f"f{width}"
    assert "no_inline" in result.mlir_text and "llvm." not in result.mlir_text
    for case, n in enumerate((0, 1, 13, 31)):
        arrays, expected = domain(n, width)
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{case}", python=NATIVE,
            shape_abi=result.spec.runtime_shapes())
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(value.view(getattr(np, f"uint{width}")), reference)


@NATIVE_REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_merlin_reuses_float_exchange_library_with_bit_exact_outputs(tmp_path, width, scalar):
    from test_integer_products import replay_library
    result = capture(tmp_path, width, scalar, order="acq_rel", scope="sys")
    fp = getattr(np, f"float{width}")
    cases = [(arrays, [a.view(fp) for a in expected]) for arrays, expected
        in (domain(n, width) for n in (0, 1, 13, 31))]
    replay_library(tmp_path, result, cases)
    for i, (_, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as actual:
            for j, reference in enumerate(expected):
                np.testing.assert_array_equal(actual[f"out{j}"].view(getattr(np, f"uint{width}")),
                    reference.view(getattr(np, f"uint{width}")))


@NATIVE_REQUIRED
@pytest.mark.parametrize("effect_profile", ["ordinary", "atomics"])
def test_legacy_profiles_do_not_implicitly_admit_float_exchange(tmp_path, effect_profile):
    with pytest.raises(ConversionError, match="qualified"):
        capture(tmp_path, effect_profile=effect_profile)


@NATIVE_REQUIRED
def test_float_add_stays_refused_without_numeric_contract(tmp_path):
    with pytest.raises(ConversionError, match="qualified"):
        capture(tmp_path, operation="add")


def test_float_contract_is_explicit_and_operation_specific():
    from t2m.atomic_memory import contract
    from xdsl.dialects import builtin
    def source(element, kind=10, cas=False):
        return builtin.UnregisteredOp.with_name("tt.atomic_cas" if cas else "tt.atomic_rmw").create(
            result_types=[element], attributes={name: builtin.IntegerAttr(value, builtin.i32)
                for name, value in (("atomic_rmw_op", kind), ("sem", 1), ("scope", 1))})
    for element in (builtin.f32, builtin.f64):
        info = contract(source(element), lambda _: ["source"], typed=True)
        assert info["numeric_contract"] == "bitwise-exchange" and info["value_dtype"] == str(element)
        for kind in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            with pytest.raises(ConversionError, match="qualified"):
                contract(source(element, kind), lambda _: ["source"], typed=True)
        for op, typed in ((source(element), False), (source(element, cas=True), True), (source(builtin.f16), True)):
            with pytest.raises(ConversionError, match="qualified"):
                contract(op, lambda _: ["source"], typed=typed)
    legacy = contract(source(builtin.i32), lambda _: ["source"])
    assert "value_dtype" not in legacy and legacy["word_bits"] == 32


def exchange_evidence():
    from t2m.atomic_memory import TYPED_EFFECT_PROFILE, contract
    from xdsl.dialects import builtin
    op = builtin.UnregisteredOp.with_name("tt.atomic_rmw").create(result_types=[builtin.f32], attributes={
        name: builtin.IntegerAttr(value, builtin.i32) for name, value in (("atomic_rmw_op", 10), ("sem", 1), ("scope", 1))})
    info = contract(op, lambda _: ["source"], typed=True)
    access = {"buffer": "state", "source_ids": ["source"], "opcode": "tts.unstructured_atomic_rmw",
        "effect": "atomic-readwrite", "mask": "source-per-lane", "atomic": info}
    payload = {**info, "buffer": "state", "source_ids": ["source"],
        "payload": "standard-memref-atomic-rmw-with-original-returned-old-value", "effect_boundary": "standard-execute-region-no-inline",
        "inactive_old_value": "zero-refinement-of-source-undefined-old-value"}
    admission = {"schema": "t2m.memory-admission.v4", "effect_profile": TYPED_EFFECT_PROFILE,
        "address_policy": "source-words-offset64", "accesses": [access], "atomic_adaptations": [payload]}
    memory = {"effect_profile": TYPED_EFFECT_PROFILE, "atomic_payload": deepcopy(admission["atomic_adaptations"])}
    return admission, memory, [{"id": "source", "opcode": "tt.atomic_rmw", "atomic": deepcopy(info)}], {"state": {"name": "state", "dtype": "f32"}}


@pytest.mark.parametrize("change", [None, "schema", "profile", "operation", "width", "numeric", "dtype", "old", "source", "order"])
def test_float_effect_evidence_rejects_rehashed_inconsistent_declarations(change):
    from t2m.atomic_memory import validate_evidence
    admission, memory, records, buffers = exchange_evidence()
    if change is None:
        validate_evidence(admission, memory, records, buffers)
        return
    if change == "schema": admission["schema"] = "t2m.memory-admission.v3"
    elif change == "profile": memory["effect_profile"] = "private-invocation-integer-atomics-v1"
    elif change == "dtype": buffers["state"]["dtype"] = "i32"
    elif change == "old": admission["atomic_adaptations"][0]["payload"] = "discarded"
    elif change == "source": records[0]["opcode"] = "tt.atomic_cas"
    else:
        update = {"operation": {"operation": "add", "standard_kind": 1}, "width": {"word_bits": 64},
            "numeric": {"numeric_contract": "rounding-ignored"}, "order": {"source_order": "unordered"}}[change]
        admission["accesses"][0]["atomic"].update(update)
        records[0]["atomic"].update(update)
        admission["atomic_adaptations"][0].update(update)
    memory["atomic_payload"] = deepcopy(admission["atomic_adaptations"])
    with pytest.raises(ValueError, match="atomic memory"):
        validate_evidence(admission, memory, records, buffers)


@NATIVE_REQUIRED
def test_float_exchange_demo_produces_verified_package(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.cli import shared_demo
    out = tmp_path / "demo"
    shared_demo(out, kernel="atomic-exchange", optimizer=OPT, python=NATIVE,
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile="typed-atomics")
    verify_bundle(out)
    compatibility = json.loads((out / "compatibility.json").read_text())
    assert "bitwise f32/f64 exchange" in json.dumps(compatibility)


@NATIVE_REQUIRED
@pytest.mark.parametrize("width", [32, 64])
def test_colliding_float_exchange_returns_one_complete_word_history(tmp_path, width):
    from t2m.native import execute_native
    result = capture(tmp_path, width)
    uint, fp = getattr(np, f"uint{width}"), getattr(np, f"float{width}")
    nan = 0x7f800001 if width == 32 else 0x7ff0000000000001
    words = np.arange(31, dtype=uint) + uint(nan)
    incoming = np.zeros(37, dtype=uint)
    incoming[0] = uint(1 << (width - 1))
    index = np.zeros(31, dtype=np.int32)
    active = np.arange(31) % 3 != 1
    index[~active] = -1000
    arrays = [words.view(fp), index, incoming.view(fp), np.asarray(31, np.int32), np.asarray(37, np.int32)]
    (state, old), _ = execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE,
        shape_abi=result.spec.runtime_shapes())
    state, old = state.view(uint), old.view(uint)
    np.testing.assert_array_equal(old[~active], np.zeros(np.count_nonzero(~active), dtype=uint))
    np.testing.assert_array_equal(state[1:], incoming[1:])
    edges = {int(before): int(after) for before, after in zip(old[active], words[active], strict=True)}
    assert len(edges) == np.count_nonzero(active)
    current = int(incoming[0])
    for _ in range(len(edges)):
        current = edges.pop(current)
    assert not edges and current == int(state[0])


@NATIVE_REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("operation", ["add", "cas"])
def test_typed_profile_retains_existing_integer_memory_semantics(tmp_path, operation):
    from test_integer_products import replay_library
    if operation == "cas":
        from test_compare_exchange import capture as capture_cas, cases
        result = capture_cas(tmp_path, profile="typed-atomics")
        domains = cases(32)
    else:
        from test_atomic_memory import capture as capture_rmw, domains as integer_domains
        from t2m.upstream import inspect_optimizer
        result = capture_rmw(tmp_path, inspect_optimizer(OPT), effect_profile="typed-atomics")
        domains = integer_domains("add", 32)
    replay_library(tmp_path, result, domains)
    differences = json.loads((tmp_path / "capture/compatibility.json").read_text())["differences"]
    if operation == "cas":
        assert any("compare/exchange" in item for item in differences)
    assert json.loads(result.auxiliary_artifacts["memory-admission.json"])["schema"] == "t2m.memory-admission.v4"
