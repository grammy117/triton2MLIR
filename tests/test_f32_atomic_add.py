"""F32 FADD retains sign-preserving FTZ as actual executable dataflow."""

import os

import numpy as np
import pytest

from test_float_atomics import MERLIN, NATIVE, OPT, capture
from test_float_atomic_add import check

REQUIRED = pytest.mark.skipif(not (OPT and NATIVE and os.environ.get("T2M_MERLIN_ROOT")),
    reason="existing shared/native/explicit Merlin support required")


def flush_word(word):
    word = int(word)
    return np.uint32(word & 0x80000000 if word & 0x7f800000 == 0 else word)


def domains(policy="ftz"):
    states = np.array([1, 0x80000001, 0x00800000, 0x00800000, 0x80800000,
        0x80000000, 0, 0x3f800000, 0x3f800001, 0x7f7fffff, 0x7f800000,
        0x7f801234, 0x3f800000, 0xffc05678], np.uint32)
    values = np.array([1, 0x80000001, 0x807fffff, 0x80800001, 0x00800001,
        0x80000000, 0x80000000, 0x33800000, 0x33800000, 0x7f7fffff, 0xff800000,
        0x3f800000, 0x7f801234, 0xffc01234], np.uint32)
    for n, rotation in ((0, 0), (1, 0), (13, 0), (31, 0), (31, 1), (31, 2)):
        incoming = np.resize(np.roll(states, rotation), 37)
        inputs = np.resize(np.roll(values, rotation), n)
        index = np.arange(n, dtype=np.int32)
        active = index % 3 != 1
        index[~active] = -1000
        updated, old = incoming.copy(), np.zeros(n, np.uint32)
        for lane in range(n):
            if active[lane]:
                old[lane] = updated[lane]
                lhs, rhs = (flush_word(updated[lane]), flush_word(inputs[lane])) if policy == "ftz" else (updated[lane], inputs[lane])
                with np.errstate(invalid="ignore", over="ignore"):
                    word = np.add(lhs.view(np.float32), rhs.view(np.float32)).view(np.uint32)
                    updated[lane] = flush_word(word) if policy == "ftz" else word
        yield [inputs.view(np.float32), index, incoming.view(np.float32), np.asarray(n, np.int32), np.asarray(37, np.int32)], \
              [updated.view(np.float32), old.view(np.float32)]


@REQUIRED
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("policy", ["ftz", "ieee"])
def test_native_f32_add_preserves_selected_policy_with_exact_old_words(tmp_path, scalar, policy):
    from t2m.native import execute_native
    result = capture(tmp_path, 32, scalar, operation="add", f32_atomic_add=policy)
    info = result.trace["memory"]["atomic_payload"][0]
    assert info["numeric_contract"] == ("f32-rne-sign-preserving-ftz-add" if policy == "ftz" else "ieee-f32-rne-add")
    assert "memref.generic_atomic_rmw" in result.mlir_text
    assert "llvm." not in result.mlir_text
    for i, (arrays, expected) in enumerate(domains(policy)):
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
            shape_abi=result.spec.runtime_shapes())
        check(actual, expected, arrays, 32)


@REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("policy", ["ftz", "ieee"])
def test_merlin_f32_add_reuses_library_across_subnormal_and_special_values(tmp_path, scalar, policy):
    from test_integer_products import replay_library
    result = capture(tmp_path, 32, scalar, operation="add", order="acq_rel", scope="sys", f32_atomic_add=policy)
    cases = list(domains(policy))
    replay_library(tmp_path, result, cases)
    for i, (arrays, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            check([outputs[f"out{j}"] for j in range(2)], expected, arrays, 32)


def evidence(policy):
    from copy import deepcopy
    from test_float_atomics import exchange_evidence
    from t2m.atomic_float_add import PAYLOAD, IEEE_PAYLOAD
    admission, memory, records, buffers = exchange_evidence()
    admission["f32_atomic_add"] = memory["f32_atomic_add"] = policy
    info = admission["accesses"][0]["atomic"]
    info.update(operation="fadd", standard_kind=0,
        numeric_contract="f32-rne-sign-preserving-ftz-add" if policy == "ftz" else "ieee-f32-rne-add")
    admission["atomic_adaptations"][0].update(info)
    admission["atomic_adaptations"][0]["payload"] = PAYLOAD if policy == "ftz" else IEEE_PAYLOAD
    records[0]["atomic"] = deepcopy(info)
    memory["atomic_payload"] = deepcopy(admission["atomic_adaptations"])
    return admission, memory, records, buffers


@pytest.mark.parametrize("policy", ["ieee", "ftz"])
@pytest.mark.parametrize("change", [None, "missing-policy", "wrong-policy", "invalid-policy", "width", "numeric", "payload", "legacy"])
def test_f32_add_evidence_binds_selected_policy_not_just_rehashed_files(policy, change):
    from copy import deepcopy
    from t2m.atomic_memory import validate_evidence
    admission, memory, records, buffers = evidence(policy)
    info = admission["accesses"][0]["atomic"]
    if change == "missing-policy":
        admission.pop("f32_atomic_add")
        memory.pop("f32_atomic_add")
    elif change in ("wrong-policy", "invalid-policy"):
        admission["f32_atomic_add"] = memory["f32_atomic_add"] = ["ieee"] if change == "invalid-policy" else "ftz" if policy == "ieee" else "ieee"
    elif change == "width":
        info.update(value_dtype="f64", word_bits=64)
        buffers["state"]["dtype"] = "f64"
    elif change == "numeric": info["numeric_contract"] = "rounding-ignored"
    elif change == "payload": admission["atomic_adaptations"][0]["payload"] = "other-numeric-payload"
    elif change == "legacy":
        admission.update(schema="t2m.memory-admission.v3", effect_profile="private-invocation-integer-atomics-v1")
        memory["effect_profile"] = admission["effect_profile"]
    records[0]["atomic"] = deepcopy(info)
    admission["atomic_adaptations"][0].update(info)
    memory["atomic_payload"] = deepcopy(admission["atomic_adaptations"])
    if change is None:
        validate_evidence(admission, memory, records, buffers)
    else:
        with pytest.raises(ValueError, match="atomic memory"):
            validate_evidence(admission, memory, records, buffers)


@REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("policy", ["ftz", "ieee"])
@pytest.mark.parametrize("scalar,while_,select", [(False, False, "none"), (True, True, "uniform"),
    (False, True, "lane"), (True, False, "lane")])
def test_merlin_f32_add_preserves_regions_selection_and_pointer_recurrence(tmp_path, policy, scalar, while_, select):
    from test_float_atomic_add import region, region_domains
    from test_atomic_word_views import exact
    from test_integer_products import replay_library
    result = region(tmp_path, scalar, while_, select, width=32, f32_atomic_add=policy)
    cases = [([a.astype(np.float32) if a.dtype == np.float64 else a for a in arrays],
              [a.astype(np.float32) for a in expected]) for arrays, expected in region_domains(select)]
    replay_library(tmp_path, result, cases)
    for i, (_, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            exact([outputs[f"out{j}"] for j in range(3)], expected, 32)


@REQUIRED
@pytest.mark.parametrize("policy", ["ieee", "ftz"])
def test_f32_add_demo_keeps_selected_policy_in_verified_package(tmp_path, policy):
    from t2m.bundle import verify_bundle
    from t2m.cli import shared_demo
    out = tmp_path / "capture"
    shared_demo(out, kernel="atomic-fadd32", optimizer=OPT, python=NATIVE,
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile="typed-atomics", f32_atomic_add=policy)
    receipt = verify_bundle(out)
    assert receipt["materialized_abi"]["outputs"] == 2
    assert ("f32-rne-sign-preserving-ftz-add" if policy == "ftz" else "ieee-f32-rne-add") in (out / "model.mlir").read_text()
    assert ("subnormal inputs/results to sign-preserving zero" if policy == "ftz" else "IEEE f32 atomic addition") in (out / "compatibility.json").read_text()


@REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("policy", ["ftz", "ieee"])
@pytest.mark.parametrize("word", [0, 0x80000000, 1, 0x7f801234])
def test_constant_f32_add_keeps_numeric_policy_after_canonicalization(tmp_path, policy, word):
    from t2m.examples.atomic_kernels import constant_fadd32
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from test_integer_products import replay_library
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(constant_fadd32, {"state": "*fp32", "old": "*fp32", "B": "constexpr", "WORD": "constexpr"}, {"B": 16, "WORD": word}))
    abi = TensorABI("constant_fadd32", (TensorBuffer("state", (16,), "readwrite", dtype="f32"),
        TensorBuffer("old", (16,), "write", dtype="f32")), grid=(1,))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics", f32_atomic_add=policy)
    incoming = list(domains())[3][0][2][:16].copy()
    updated = incoming.view(np.uint32).copy()
    for i, old in enumerate(updated):
        lhs, rhs = (flush_word(old), flush_word(word)) if policy == "ftz" else (old, np.uint32(word))
        with np.errstate(invalid="ignore", over="ignore"):
            expected = np.add(lhs.view(np.float32), rhs.view(np.float32)).view(np.uint32)
        updated[i] = flush_word(expected) if policy == "ftz" else expected
    expected = [updated.view(np.float32), incoming.copy()]
    replay_library(tmp_path, result, [([incoming], expected)])
    with np.load(tmp_path / "invoke0/evaluation/outputs.npz", allow_pickle=False) as outputs:
        actual, nan = outputs["out0"], np.isnan(expected[0])
        np.testing.assert_array_equal(np.isnan(actual), nan)
        np.testing.assert_array_equal(actual[~nan].view(np.uint32), updated[~nan])
        assert np.all(actual[nan].view(np.uint32) & np.uint32(1 << 22))
        np.testing.assert_array_equal(outputs["out1"].view(np.uint32), incoming.view(np.uint32))
