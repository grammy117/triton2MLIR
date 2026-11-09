"""Source word atomics and floating min/max keep exact representations."""

from copy import deepcopy
import json
import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
REQUIRED = pytest.mark.skipif(not (OPT and NATIVE and os.environ.get("T2M_MERLIN_ROOT")), reason="existing shared/native and explicit Merlin compiler support required")
OPS = ("add", "and", "or", "xor", "max", "min", "umax", "umin", "exch")


def word_capture(root, width, operation, *, scalar=False, effect_profile="typed-atomics"):
    from t2m.examples.atomic_kernels import indexed_word_atomic
    from t2m.frontend import capture_ttir
    constants = {"B": 8, "WIDTH": width, "OP": operation, "SCALAR": scalar}
    source = root / "source.ttir"
    source.write_text(capture_ttir(indexed_word_atomic, {"x": f"*fp{width}", "index": "*i32", "state": f"*fp{width}", "old": f"*fp{width}",
        "n": "i32", "m": "i32", **{n: "constexpr" for n in constants}}, constants))
    abi = TensorABI("indexed_word_atomic", (TensorBuffer("x", ("n",), dtype=f"f{width}"), TensorBuffer("index", ("n",), dtype="i32"),
        TensorBuffer("state", ("m",), "readwrite", dtype=f"f{width}"), TensorBuffer("old", ("n",), "write", dtype=f"f{width}")),
        grid=(GridExtent("n", 1 if scalar else 8),), runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
        RuntimeScalar("m", "i32", 5, bounds=(0, 37))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile=effect_profile)


def domains(width, operation, *, floating=False):
    from test_atomic_memory import operation_reference, signed
    from test_float_atomics import domain
    uint, fp = getattr(np, f"uint{width}"), getattr(np, f"float{width}")
    for n in (0, 1, 13, 31):
        arrays, _ = domain(n, width)
        values, index, initial = arrays[:3]
        state, old = initial.view(uint).copy(), np.zeros(n, dtype=uint)
        for lane in range(n):
            if lane % 3 == 1:
                continue
            at, word = int(index[lane]), int(values.view(uint)[lane])
            old[lane] = state[at]
            kind = operation
            if floating:
                # Independent integer-word oracle for Triton's sign-conditioned
                # source algorithm, including signaling NaNs and signed zeros.
                if word & (1 << (width - 1)):
                    kind = "umax" if operation == "min" else "umin"
            result = operation_reference(signed(int(state[at]), width), signed(word, width), kind, width)
            state[at] = uint(result & ((1 << width) - 1))
        yield arrays, [state.view(fp), old.view(fp)]


def exact(actual, expected, width):
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a.view(getattr(np, f"uint{width}")), b.view(getattr(np, f"uint{width}")))


@REQUIRED
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("operation", OPS)
def test_native_word_updates_preserve_float_storage_and_returned_bits(tmp_path, width, operation):
    from t2m.native import execute_native
    result = word_capture(tmp_path, width, operation)
    info = result.trace["memory"]["atomic_payload"][0]
    assert info["operation"] == operation
    assert info["word_view"] == {"storage_dtype": f"f{width}", "value_dtype": f"i{width}", "element_bits": width, "address_units": "equal-width-elements"}
    assert "memref.generic_atomic_rmw" in result.mlir_text and "prov.logical_word_views" in result.mlir_text
    assert "llvm." not in result.mlir_text
    for i, (arrays, expected) in enumerate(domains(width, operation)):
        actual, report = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
            shape_abi=result.spec.runtime_shapes())
        exact(actual, expected, width)
        assert report["compiler_adapters"][0]["name"] == "merlin-exact-atomic-word-views"


@REQUIRED
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("operation", ["min", "max"])
@pytest.mark.parametrize("scalar", [False, True])
def test_native_floating_minmax_retains_source_nan_zero_and_subnormal_behavior(tmp_path, width, operation, scalar):
    from test_float_atomics import capture
    from t2m.native import execute_native
    result = capture(tmp_path, width, scalar, operation=operation)
    assert len(result.trace["memory"]["atomic_payload"]) == 2
    for i, (arrays, expected) in enumerate(domains(width, operation, floating=True)):
        actual, report = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
            shape_abi=result.spec.runtime_shapes())
        exact(actual, expected, width)
        assert report["compiler_adapters"][0]["events"] == 2


@REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("operation", OPS)
def test_merlin_word_updates_reuse_the_library_with_exact_outputs(tmp_path, width, operation):
    from test_integer_products import replay_library
    result = word_capture(tmp_path, width, operation)
    cases = list(domains(width, operation))
    replay_library(tmp_path, result, cases)
    for i, (_, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            exact([outputs[f"out{j}"] for j in range(2)], expected, width)
    report = json.loads((tmp_path / "merlin/host/compilation.json").read_text())
    assert "lower_atomic_word_views" in report["features"]
    llvm = (tmp_path / "merlin/host/model.ll").read_text()
    assert "define internal" in llvm and "atomicrmw" in llvm


@REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("operation", ["min", "max"])
def test_merlin_floating_minmax_keeps_both_original_word_atomics(tmp_path, width, operation):
    from test_float_atomics import capture
    from test_integer_products import replay_library
    result = capture(tmp_path, width, operation=operation)
    cases = list(domains(width, operation, floating=True))
    replay_library(tmp_path, result, cases)
    for i, (_, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            exact([outputs[f"out{j}"] for j in range(2)], expected, width)


def word_evidence():
    from test_float_atomics import exchange_evidence
    from t2m.atomic_word_views import PAYLOAD
    admission, memory, records, buffers = exchange_evidence()
    info = admission["accesses"][0]["atomic"]
    info.pop("value_dtype")
    info.pop("numeric_contract")
    info.update(operation="max", standard_kind=6, word_view={"storage_dtype": "f32", "value_dtype": "i32",
        "element_bits": 32, "address_units": "equal-width-elements"})
    records[0]["atomic"] = deepcopy(info)
    payload = admission["atomic_adaptations"][0]
    payload.pop("value_dtype")
    payload.pop("numeric_contract")
    payload.update(info, payload=PAYLOAD)
    memory["atomic_payload"] = deepcopy(admission["atomic_adaptations"])
    return admission, memory, records, buffers


@pytest.mark.parametrize("change", [None, "profile", "dtype", "source", "lost-view", "storage", "value", "units", "bits", "kind", "old"])
def test_word_evidence_rejects_rehashed_inconsistent_contracts(change):
    from t2m.atomic_memory import validate_evidence
    admission, memory, records, buffers = word_evidence()
    info = admission["accesses"][0]["atomic"]
    if change is None:
        validate_evidence(admission, memory, records, buffers)
        return
    if change == "profile":
        admission.update(schema="t2m.memory-admission.v3", effect_profile="private-invocation-integer-atomics-v1")
        memory["effect_profile"] = admission["effect_profile"]
    elif change == "dtype": buffers["state"]["dtype"] = "i32"
    elif change == "source": records[0]["opcode"] = "tt.atomic_cas"
    elif change == "old": admission["atomic_adaptations"][0]["payload"] = "discarded"
    else:
        if change == "lost-view": info.pop("word_view")
        elif change == "bits": info["word_bits"] = 64
        elif change == "kind": info["standard_kind"] = 42
        else:
            key, value = {"storage": ("storage_dtype", "f64"), "value": ("value_dtype", "f32"), "units": ("address_units", "bytes")}[change]
            info["word_view"][key] = value
        records[0]["atomic"] = deepcopy(info)
        payload = admission["atomic_adaptations"][0]
        payload.pop("word_view", None)
        payload.update(info)
    memory["atomic_payload"] = deepcopy(admission["atomic_adaptations"])
    with pytest.raises(ValueError, match="atomic memory"):
        validate_evidence(admission, memory, records, buffers)


@REQUIRED
@pytest.mark.parametrize("operation", ["min", "max"])
def test_floating_minmax_demo_retains_the_standard_merlin_package(tmp_path, operation):
    from t2m.bundle import verify_bundle
    from t2m.cli import shared_demo
    out = tmp_path / "capture"
    shared_demo(out, kernel=f"atomic-{operation}", optimizer=OPT, python=NATIVE,
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile="typed-atomics")
    receipt = verify_bundle(out)
    assert receipt["materialized_abi"]["outputs"] == 2
    assert "memref.generic_atomic_rmw" in (out / "model.mlir").read_text()
    assert "integer word atomics" in (out / "compatibility.json").read_text()


@REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("operation", ["add", "and", "max", "min", "exch"])
def test_merlin_preserves_constant_identity_and_assignment_atomics(tmp_path, width, operation):
    from t2m.examples.atomic_kernels import constant_word_atomic
    from t2m.frontend import capture_ttir
    from test_atomic_memory import operation_reference, signed
    from test_integer_products import replay_library
    value = {"add": 0, "and": -1, "max": (1 << (width - 1)) - 1, "min": -(1 << (width - 1)),
        "exch": 0x7f801234 if width == 32 else 0x7ff0000000001234}[operation]
    constants = {"B": 8, "WIDTH": width, "OP": operation, "VALUE": value}
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(constant_word_atomic, {"state": f"*fp{width}", "old": f"*fp{width}",
        **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("constant_word_atomic", (TensorBuffer("state", (8,), "readwrite", dtype=f"f{width}"),
        TensorBuffer("old", (8,), "write", dtype=f"f{width}")))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    uint, fp = getattr(np, f"uint{width}"), getattr(np, f"float{width}")
    maximum, sign = (1 << width) - 1, 1 << (width - 1)
    samples = np.array([0, sign, 1, sign + 1, maximum, value & maximum, 0x12345678, 0x7f801234], dtype=uint)
    cases = []
    for incoming in (samples, samples[::-1].copy()):
        updated = np.array([operation_reference(signed(int(word), width), value, operation, width) & maximum
            for word in incoming], dtype=uint)
        cases.append(([incoming.view(fp)], [updated.view(fp), incoming.view(fp)]))
    replay_library(tmp_path, result, cases)
    for i, (_, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            exact([outputs[f"out{j}"] for j in range(2)], expected, width)
