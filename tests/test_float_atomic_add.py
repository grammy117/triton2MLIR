"""IEEE f64 atomic addition: exact finite results, old words and NaN classes."""

from copy import deepcopy

import numpy as np
import pytest

from test_float_atomics import NATIVE_REQUIRED, MERLIN, NATIVE, capture, exchange_evidence


def domains():
    tiny = np.finfo(np.float64).tiny
    sub = np.nextafter(np.float64(0), np.float64(1))
    maximum = np.finfo(np.float64).max
    states = np.array([1, np.nextafter(1., 2.), tiny, sub, tiny - sub,
                       -0., 0., maximum, np.inf, np.nan, 1., -sub], np.float64)
    values = np.array([2.**-53, 2.**-53, -(tiny - sub), sub, sub,
                       -0., -0., maximum, -np.inf, 1., np.nan, -sub], np.float64)
    # A signaling input must survive unchanged in the returned old value.
    states.view(np.uint64)[9] = 0x7ff0000000001234
    for n, rotation in ((0, 0), (1, 0), (13, 0), (31, 0), (31, 1), (31, 2)):
        incoming = np.resize(np.roll(states, rotation), 37)
        inputs = np.resize(np.roll(values, rotation), n)
        index = np.arange(n, dtype=np.int32)
        active = index % 3 != 1
        index[~active] = -1000
        updated, old = incoming.copy(), np.zeros(n, np.float64)
        for lane in range(n):
            if active[lane]:
                old.view(np.uint64)[lane] = updated.view(np.uint64)[lane]
                with np.errstate(invalid="ignore", over="ignore"):
                    updated[lane] = np.add(updated[lane], inputs[lane])
        yield [inputs, index, incoming, np.asarray(n, np.int32), np.asarray(37, np.int32)], [updated, old]


def check(actual, expected, arrays, width=64):
    uint = getattr(np, f"uint{width}")
    updated, old = actual
    state, previous = expected
    # Arithmetic NaN payload/sign is unspecified; finite values and zeros are exact.
    nan = np.isnan(state)
    np.testing.assert_array_equal(np.isnan(updated), nan)
    np.testing.assert_array_equal(updated[~nan].view(uint), state[~nan].view(uint))
    n, index = len(arrays[0]), arrays[1]
    written = np.zeros(state.shape, np.bool_)
    written[index[np.arange(n) % 3 != 1]] = True
    assert np.all(updated[nan & written].view(uint) & uint(1 << (51 if width == 64 else 22)))
    np.testing.assert_array_equal(updated[~written].view(uint), state[~written].view(uint))
    np.testing.assert_array_equal(old.view(uint), previous.view(uint))


@NATIVE_REQUIRED
@pytest.mark.parametrize("scalar", [False, True])
def test_native_f64_add_keeps_rounding_subnormals_and_old_words(tmp_path, scalar):
    from t2m.native import execute_native
    result = capture(tmp_path, 64, scalar, operation="add")
    info = result.trace["memory"]["atomic_payload"][0]
    assert info["operation"] == "fadd" and info["numeric_contract"] == "ieee-f64-rne-add"
    assert "prov.atomic_numeric_contract" in result.mlir_text
    for i, (arrays, expected) in enumerate(domains()):
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
            shape_abi=result.spec.runtime_shapes())
        check(actual, expected, arrays)


@NATIVE_REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("scalar", [False, True])
def test_merlin_f64_add_reuses_library_across_rounding_and_special_values(tmp_path, scalar):
    from test_integer_products import replay_library
    result = capture(tmp_path, 64, scalar, operation="add", order="acq_rel", scope="sys")
    cases = list(domains())
    replay_library(tmp_path, result, cases)
    for i, (arrays, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            check([outputs[f"out{j}"] for j in range(2)], expected, arrays)


def evidence():
    admission, memory, records, buffers = exchange_evidence()
    info = admission["accesses"][0]["atomic"]
    info.update(operation="fadd", standard_kind=0, word_bits=64, value_dtype="f64",
                numeric_contract="ieee-f64-rne-add")
    admission["atomic_adaptations"][0].update(info)
    memory["atomic_payload"] = deepcopy(admission["atomic_adaptations"])
    records[0]["atomic"] = deepcopy(info)
    buffers["state"]["dtype"] = "f64"
    return admission, memory, records, buffers


@pytest.mark.parametrize("change", [None, "f32", "integer", "ftz", "rounding", "kind", "legacy"])
def test_f64_add_evidence_cannot_erase_or_change_numeric_policy(change):
    from t2m.atomic_memory import validate_evidence
    admission, memory, records, buffers = evidence()
    info = admission["accesses"][0]["atomic"]
    if change == "f32":
        info.update(word_bits=32, value_dtype="f32")
        buffers["state"]["dtype"] = "f32"
    elif change == "integer":
        info.pop("value_dtype")
        info.pop("numeric_contract")
        buffers["state"]["dtype"] = "i64"
    elif change in ("ftz", "rounding"): info["numeric_contract"] = change
    elif change == "kind": info["standard_kind"] = 1
    elif change == "legacy":
        admission.update(schema="t2m.memory-admission.v3", effect_profile="private-invocation-integer-atomics-v1")
        memory["effect_profile"] = admission["effect_profile"]
    records[0]["atomic"] = deepcopy(info)
    admission["atomic_adaptations"][0] = {**admission["atomic_adaptations"][0], **info}
    if change == "integer":
        admission["atomic_adaptations"][0].pop("value_dtype")
        admission["atomic_adaptations"][0].pop("numeric_contract")
    memory["atomic_payload"] = deepcopy(admission["atomic_adaptations"])
    if change is None:
        validate_evidence(admission, memory, records, buffers)
    else:
        with pytest.raises(ValueError, match="atomic memory"):
            validate_evidence(admission, memory, records, buffers)


def region(root, scalar, while_, select, *, width=64, f32_atomic_add=None):
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.atomic_kernels import region_fadd
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from test_float_atomics import OPT
    constants = {"B": 8, "SCALAR": scalar, "WHILE": while_, "SELECT": select}
    source = root / "source.ttir"
    source.write_text(capture_ttir(region_fadd, {"a": f"*fp{width}", "b": f"*fp{width}", "out": f"*fp{width}",
        "n": "i32", "steps": "i32", "enabled": "i1", "choice": "i1", "shift": "i32",
        **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("region_fadd", (TensorBuffer("a", (4, "n"), "readwrite", dtype=f"f{width}"),
        TensorBuffer("b", (4, "n"), "readwrite", dtype=f"f{width}"), TensorBuffer("out", ("n",), "write", dtype=f"f{width}")),
        grid=(GridExtent("n", 1 if scalar else 8),), runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 13)),
        RuntimeScalar("steps", "i32", 4, bounds=(0, 3)), RuntimeScalar("enabled", "i1", 5),
        RuntimeScalar("choice", "i1", 6), RuntimeScalar("shift", "i32", 7, bounds=(-1000, 1000))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics", f32_atomic_add=f32_atomic_add)


def region_domains(select):
    for n, steps, enabled, choice, shift in ((0, 3, True, False, 0), (1, 0, True, True, 0),
            (9, 2, True, False, 0), (13, 3, True, True, 0), (9, 3, False, True, 1000)):
        a = np.arange(4 * n, dtype=np.float64).reshape(4, n) - 19
        b = a[::-1].copy() + 7
        first, second, out = a.copy(), b.copy(), np.zeros(n, np.float64)
        if enabled:
            for lane in range(n):
                target = second if select == "lane" and lane % 2 or select == "uniform" and choice else first
                total = np.float64(0)
                for row in range(steps):
                    previous = target[row, lane]
                    target[row, lane] += row + 1
                    total += previous
                previous = target[steps, lane]
                target[steps, lane] += total
                out[lane] = total + previous
        arrays = [a, b, np.asarray(n, np.int32), np.asarray(steps, np.int32), np.asarray(enabled, np.bool_),
                  np.asarray(choice, np.bool_), np.asarray(shift, np.int32)]
        yield arrays, [first, second, out]


@NATIVE_REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("scalar,while_,select", [(False, False, "none"), (True, True, "none"),
    (False, True, "uniform"), (True, False, "uniform"), (False, True, "lane"), (False, False, "lane")])
def test_merlin_f64_add_preserves_regions_and_selected_pointer_recurrences(tmp_path, scalar, while_, select):
    from test_atomic_word_views import exact
    from test_integer_products import replay_library
    result = region(tmp_path, scalar, while_, select)
    cases = list(region_domains(select))
    replay_library(tmp_path, result, cases)
    for i, (_, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            exact([outputs[f"out{j}"] for j in range(3)], expected, 64)


@NATIVE_REQUIRED
def test_f64_add_demo_keeps_numeric_contract_in_verified_package(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.cli import shared_demo
    from test_float_atomics import OPT
    out = tmp_path / "capture"
    shared_demo(out, kernel="atomic-fadd", optimizer=OPT, python=NATIVE,
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile="typed-atomics")
    receipt = verify_bundle(out)
    assert receipt["materialized_abi"]["outputs"] == 2
    assert "ieee-f64-rne-add" in (out / "model.mlir").read_text()
    assert "IEEE f64" in (out / "compatibility.json").read_text()


@NATIVE_REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
def test_colliding_f64_tickets_preserve_one_atomic_history_without_lane_order(tmp_path):
    import os
    from t2m.bundle import _record
    from t2m.consumer import invoke_merlin_host
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.atomic_kernels import atomic_tickets
    from t2m.frontend import capture_ttir
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_bundle import write_shared_bundle
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from test_float_atomics import OPT
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(atomic_tickets, {"state": "*fp64", "out": "*fp64", "n": "i32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("atomic_tickets", (TensorBuffer("state", (1,), "readwrite", dtype="f64"),
        TensorBuffer("out", ("n",), "write", dtype="f64")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 31)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, [np.array([0.], np.float64), np.asarray(1, np.int32)],
        [np.array([1.], np.float64), np.array([0.], np.float64)], work=tmp_path / "native", python=NATIVE)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, (n, start) in enumerate(((0, 7.), (13, -11.), (31, float(2**40)))):
        inputs = tmp_path / "inputs.npz"
        np.savez(inputs, in0=np.array([start], np.float64), in1=np.asarray(n, np.int32))
        (state, tickets), evaluation = invoke_merlin_host(compiled, inputs, tmp_path / f"invoke{i}", merlin_root=root, python=python)
        np.testing.assert_array_equal(state, [start + n])
        np.testing.assert_array_equal(np.sort(tickets), start + np.arange(n, dtype=np.float64))
        assert evaluation["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library
