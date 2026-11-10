"""Logical word CAS retains floating storage and exact comparison/update/old bits."""

import json
import os
from copy import deepcopy

import numpy as np
import pytest

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
REQUIRED = pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin required")


def patterns(width):
    if width == 16:
        return np.array([0, 0x8000, 0x7c00, 0xfc00, 0x7e35, 0xfe56, 0x7c11, 0xfc22,
                         1, 0x8001, 0x7bff, 0xfbff], np.uint16)
    from test_float_atomics import domain
    return domain(31, width)[0][2].view(getattr(np, f"uint{width}"))[:12]


def domains(width):
    uint, integer, floating = (getattr(np, f"{prefix}{width}") for prefix in ("uint", "int", "float"))
    for rotation in range(3):
        state = np.resize(np.roll(patterns(width), rotation), 16)
        comparison = state.copy()
        comparison[1::3] ^= uint(1)
        comparison[:2] ^= uint(1 << (width - 1))
        desired = np.resize(np.roll(patterns(width), rotation + 3), 16)
        updated = np.where(state == comparison, desired, state)
        yield [state.view(floating), comparison.view(integer), desired.view(integer)], [updated.view(floating), state.view(floating)]


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("change", [None, "legacy-source", "profile", "dtype", "source", "mask", "payload", "storage", "units", "kind", "bits"])
def test_word_cas_evidence_binds_integer_primitive_and_floating_storage(width, change):
    from test_atomic_word_views import word_evidence
    from t2m.atomic_memory import validate_evidence
    from t2m.atomic_word_views import CAS_PAYLOAD
    admission, memory, records, buffers = word_evidence()
    access, payload = admission["accesses"][0], admission["atomic_adaptations"][0]
    info = access["atomic"]
    info.update(operation="cas", standard_kind="compare-exchange", word_bits=width,
        word_view={"storage_dtype": f"f{width}", "value_dtype": f"i{width}", "element_bits": width,
                   "address_units": "equal-width-elements"})
    access.update(opcode="tt.atomic_cas", mask="unmasked")
    buffers["state"]["dtype"] = f"f{width}"
    if change == "profile": memory["effect_profile"] = "private-invocation-integer-atomics-v1"
    elif change == "dtype": buffers["state"]["dtype"] = f"i{width}"
    elif change == "mask": access["mask"] = "source-per-lane"
    elif change == "storage": info["word_view"]["storage_dtype"] = "f64" if width != 64 else "f32"
    elif change == "units": info["word_view"]["address_units"] = "bytes"
    elif change == "kind": info["standard_kind"] = 1
    elif change == "bits": info["word_bits"] = 8
    payload.update(deepcopy(info), payload=CAS_PAYLOAD, inactive_old_value="no-inactive-lanes")
    if change == "payload": payload["payload"] = "standard-memref-generic-integer-word-update-with-original-returned-old-value"
    records[0].update(opcode="tt.atomic_rmw" if change == "source" else "tt.atomic_cas", atomic=deepcopy(info))
    if change != "legacy-source": records[0]["atomic"].pop("word_view")
    memory["atomic_payload"] = deepcopy(admission["atomic_adaptations"])
    if change in (None, "legacy-source"):
        validate_evidence(admission, memory, records, buffers)
    else:
        with pytest.raises(ValueError, match="atomic memory"):
            validate_evidence(admission, memory, records, buffers)


@REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
def test_word_cas_native_template_cannot_change_or_lose_its_storage_boundary(tmp_path, width):
    from t2m.word_evidence import check_word_stage, replay_events
    result = capture(tmp_path, width, False)
    stage = json.loads(result.auxiliary_artifacts["word-stage.json"])
    source, output = (result.auxiliary_artifacts[name] for name in ("word-input.mlir", "word-tensor.mlir"))
    replay_events(stage, source)
    for change in ("missing", "formal", "storage", "capture", "opcode"):
        bad = deepcopy(stage)
        binding = next(b for b in bad["events"]["bindings"] if b["opcode"] == "tt.atomic_cas")
        assert binding["input_types"][0] == f"!tt.ptr<f{width}>"
        assert binding["template_input_types"][0] == f"!tt.ptr<i{width}>"
        if change == "missing": binding.pop("template_input_types")
        elif change == "formal": binding["template_input_types"][0] = binding["input_types"][0]
        elif change == "storage": binding["input_types"][0] = binding["template_input_types"][0]
        elif change == "capture": binding["template_input_types"][2] = "i1"
        else: binding["opcode"] = "tt.atomic_rmw"
        check_word_stage(bad, source, output)
        with pytest.raises(ValueError, match="complete typed source event inventory"):
            replay_events(bad, source)


@REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
def test_word_cas_demo_writes_the_standard_tensor_package_with_exact_words(tmp_path, width):
    from t2m.bundle import verify_bundle
    from t2m.cli import shared_demo
    out = tmp_path / "capture"
    shared_demo(out, kernel=f"atomic-word-cas{width}", optimizer=OPT, python=NATIVE,
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile="typed-atomics")
    assert verify_bundle(out)["materialized_abi"]["outputs"] == 2
    assert "integer word atomics" in (out / "compatibility.json").read_text()
    assert "llvm." not in (out / "model.mlir").read_text()
    uint = getattr(np, f"uint{width}")
    with np.load(out / "extra.npz", allow_pickle=False) as extra, np.load(
            tmp_path / "capture.work/native/outputs.npz", allow_pickle=False) as actual:
        expected = [np.load(out / "golden.npy", allow_pickle=False), extra["out1"]]
        for index, reference in enumerate(expected):
            np.testing.assert_array_equal(actual[f"out{index}"].view(uint), reference.view(uint))


def capture(root, width, scalar):
    from t2m.examples.atomic_kernels import word_cas
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    source = root / "source.ttir"
    source.write_text(capture_ttir(word_cas, {"state": f"*fp{width}", "comparison": f"*i{width}", "value": f"*i{width}",
        "old": f"*fp{width}", "B": "constexpr", "WIDTH": "constexpr", "SCALAR": "constexpr"}, {"B": 8, "WIDTH": width, "SCALAR": scalar}))
    abi = TensorABI("word_cas", (TensorBuffer("state", (16,), "readwrite", dtype=f"f{width}"),
        TensorBuffer("comparison", (16,), dtype=f"i{width}"), TensorBuffer("value", (16,), dtype=f"i{width}"),
        TensorBuffer("old", (16,), "write", dtype=f"f{width}")), grid=(16 if scalar else 2,))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")


@REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_word_cas_preserves_success_failure_old_bits_and_storage_abi(tmp_path, width, scalar):
    from test_atomic_word_views import exact
    from test_integer_products import replay_library
    result = capture(tmp_path, width, scalar)
    cases = list(domains(width))
    replay_library(tmp_path, result, cases)
    for i, (_, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            exact([outputs[f"out{j}"] for j in range(2)], expected, width)
    info = result.trace["memory"]["atomic_payload"][0]
    assert info["operation"] == "cas" and info["word_view"]["storage_dtype"] == f"f{width}"
    assert "llvm." not in result.mlir_text and "memref.generic_atomic_rmw" in result.mlir_text
    assert json.loads(result.auxiliary_artifacts["word-stage.json"])["schema"] == "t2m.source-word-stages.v2"


@REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
def test_rank_two_word_cas_retains_flattened_event_and_original_tile(tmp_path, width):
    from t2m.examples.atomic_kernels import matrix_word_cas
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from test_integer_products import replay_library
    from test_atomic_word_views import exact
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(matrix_word_cas, {"state": f"*fp{width}", "comparison": f"*i{width}", "value": f"*i{width}",
        "old": f"*fp{width}", "WIDTH": "constexpr"}, {"WIDTH": width}))
    abi = TensorABI("matrix_word_cas", (TensorBuffer("state", (16,), "readwrite", dtype=f"f{width}"),
        TensorBuffer("comparison", (16,), dtype=f"i{width}"), TensorBuffer("value", (16,), dtype=f"i{width}"),
        TensorBuffer("old", (16,), "write", dtype=f"f{width}")), grid=(2,))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    cases = list(domains(width))
    replay_library(tmp_path, result, cases)
    for i, (_, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            exact([outputs[f"out{j}"] for j in range(2)], expected, width)


@REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("scalar,lane", [(False, False), (True, False), (False, True), (True, True)])
def test_word_cas_preserves_uniform_and_per_lane_root_selection(tmp_path, width, scalar, lane):
    from t2m.examples.atomic_kernels import selected_word_cas
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from test_integer_products import replay_library
    from test_atomic_word_views import exact
    source = tmp_path / "source.ttir"
    constants = {"B": 8, "WIDTH": width, "SCALAR": scalar, "LANE": lane}
    source.write_text(capture_ttir(selected_word_cas, {"a": f"*fp{width}", "b": f"*fp{width}", "comparison": f"*i{width}",
        "value": f"*i{width}", "old": f"*fp{width}", "choice": "i1", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("selected_word_cas", (TensorBuffer("a", (16,), "readwrite", dtype=f"f{width}"),
        TensorBuffer("b", (16,), "readwrite", dtype=f"f{width}"), TensorBuffer("comparison", (16,), dtype=f"i{width}"),
        TensorBuffer("value", (16,), dtype=f"i{width}"), TensorBuffer("old", (16,), "write", dtype=f"f{width}")),
        grid=(16 if scalar else 2,), runtime_scalars=(RuntimeScalar("choice", "i1", 5),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    uint, integer, floating = (getattr(np, f"{prefix}{width}") for prefix in ("uint", "int", "float"))
    cases = []
    for choice in (False, True):
        a, b = np.resize(patterns(width), 16), np.resize(np.roll(patterns(width), 4), 16)
        selected = np.arange(16) % 2 == 1 if lane else np.full(16, choice)
        old = np.where(selected, b, a)
        comparison = old.copy()
        comparison[1::3] ^= uint(1)
        desired = np.resize(np.roll(patterns(width), 7), 16)
        updated = np.where(old == comparison, desired, old)
        first, second = np.where(selected, a, updated), np.where(selected, updated, b)
        cases.append(([a.view(floating), b.view(floating), comparison.view(integer), desired.view(integer), np.asarray(choice, np.bool_)],
                      [first.view(floating), second.view(floating), old.view(floating)]))
    replay_library(tmp_path, result, cases)
    for i, (_, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            exact([outputs[f"out{j}"] for j in range(3)], expected, width)


@REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("identity", [False, True])
def test_constant_word_cas_and_canonical_identity_remain_atomic(tmp_path, width, identity):
    from t2m.examples.atomic_kernels import constant_word_cas
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from test_integer_products import replay_library
    from test_atomic_word_views import exact
    source = tmp_path / "source.ttir"
    constants = {"B": 16, "WIDTH": width, "COMPARISON": 0, "VALUE": 0 if identity else 1}
    source.write_text(capture_ttir(constant_word_cas, {"state": f"*fp{width}", "old": f"*fp{width}",
        **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("constant_word_cas", (TensorBuffer("state", (16,), "readwrite", dtype=f"f{width}"),
        TensorBuffer("old", (16,), "write", dtype=f"f{width}")), grid=(1,))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    uint, floating = getattr(np, f"uint{width}"), getattr(np, f"float{width}")
    incoming = np.resize(patterns(width), 16)
    updated = np.where(incoming == uint(0), uint(0 if identity else 1), incoming)
    expected = [updated.view(floating), incoming.view(floating)]
    replay_library(tmp_path, result, [([incoming.view(floating)], expected)])
    with np.load(tmp_path / "invoke0/evaluation/outputs.npz", allow_pickle=False) as outputs:
        exact([outputs[f"out{j}"] for j in range(2)], expected, width)


@REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("scalar,while_", [(False, False), (True, True), (False, True), (True, False)])
def test_word_cas_keeps_pointer_and_old_value_recurrences_in_source_regions(tmp_path, width, scalar, while_):
    from t2m.dynamic_abi import GridExtent
    from t2m.examples.atomic_kernels import region_word_cas
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from test_integer_products import replay_library
    from test_atomic_word_views import exact
    source = tmp_path / "source.ttir"
    constants = {"B": 8, "WIDTH": width, "SCALAR": scalar, "WHILE": while_}
    source.write_text(capture_ttir(region_word_cas, {"state": f"*fp{width}", "out": f"*fp{width}", "n": "i32", "steps": "i32",
        "enabled": "i1", "shift": "i32", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("region_word_cas", (TensorBuffer("state", (4, "n"), "readwrite", dtype=f"f{width}"),
        TensorBuffer("out", ("n",), "write", dtype=f"f{width}")), grid=(GridExtent("n", 1 if scalar else 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 16)), RuntimeScalar("steps", "i32", 3, bounds=(0, 3)),
                        RuntimeScalar("enabled", "i1", 4), RuntimeScalar("shift", "i32", 5, bounds=(-1000, 1000))))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    uint, floating, mask = getattr(np, f"uint{width}"), getattr(np, f"float{width}"), (1 << width) - 1
    cases = []
    for n, steps, enabled, shift in ((0, 3, True, 0), (1 if scalar else 8, 0, True, 0),
            (9 if scalar else 16, 2, True, 0), (13 if scalar else 8, 3, True, 0), (9 if scalar else 16, 3, False, 1000)):
        incoming = np.resize(patterns(width), (4, n))
        for row in range(3):
            incoming[row, ::2] = uint((row - 11) & mask)
        updated, out = incoming.copy(), np.zeros(n, uint)
        if enabled:
            for lane in range(n):
                total = 0
                for row in range(steps):
                    previous = int(updated[row, lane])
                    if previous == (row - 11) & mask:
                        updated[row, lane] = uint(row + 1)
                    total = (total + previous) & mask
                previous = int(updated[steps, lane])
                if previous == total:
                    updated[steps, lane] = uint((total + 7) & mask)
                out[lane] = uint((total + previous) & mask)
        arrays = [incoming.view(floating), np.asarray(n, np.int32), np.asarray(steps, np.int32),
                  np.asarray(enabled, np.bool_), np.asarray(shift, np.int32)]
        cases.append((arrays, [updated.view(floating), out.view(floating)]))
    replay_library(tmp_path, result, cases)
    for i, (_, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            exact([outputs[f"out{j}"] for j in range(2)], expected, width)


@REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
def test_one_cas_helper_keeps_integer_and_float_storage_in_distinct_call_contexts(tmp_path, width):
    from t2m.examples.atomic_kernels import mixed_storage_cas
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from test_atomic_word_views import exact
    from test_integer_products import replay_library
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(mixed_storage_cas, {"integer_state": f"*i{width}", "floating_state": f"*fp{width}",
        "comparison": f"*i{width}", "value": f"*i{width}", "old_integer": f"*i{width}", "old_word": f"*fp{width}",
        "B": "constexpr", "WIDTH": "constexpr"}, {"B": 8, "WIDTH": width}))
    abi = TensorABI("mixed_storage_cas", (TensorBuffer("integer_state", (16,), "readwrite", dtype=f"i{width}"),
        TensorBuffer("floating_state", (16,), "readwrite", dtype=f"f{width}"), TensorBuffer("comparison", (16,), dtype=f"i{width}"),
        TensorBuffer("value", (16,), dtype=f"i{width}"), TensorBuffer("old_integer", (16,), "write", dtype=f"i{width}"),
        TensorBuffer("old_word", (16,), "write", dtype=f"f{width}")), grid=(2,))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    integer = getattr(np, f"int{width}")
    cases = []
    for (state, comparison, desired), (updated, _) in domains(width):
        incoming = np.roll(state.view(integer), 5).copy()
        updated_integer = np.where(incoming == comparison, desired, incoming)
        cases.append(([incoming, state, comparison, desired], [updated_integer, updated, incoming.copy(), state.copy()]))
    replay_library(tmp_path, result, cases)
    for path, expected in [(tmp_path / "native/outputs.npz", cases[-1][1]),
                           *[(tmp_path / f"invoke{i}/evaluation/outputs.npz", values) for i, (_, values) in enumerate(cases)]]:
        with np.load(path, allow_pickle=False) as outputs:
            exact([outputs[f"out{j}"] for j in range(4)], expected, width)
    payload = result.trace["memory"]["atomic_payload"]
    plain = next(event for event in payload if event["buffer"] == "integer_state")
    viewed = next(event for event in payload if event["buffer"] == "floating_state")
    assert set(plain["source_ids"]) & set(viewed["source_ids"])
    assert set(plain["source_ids"]) != set(viewed["source_ids"])
    assert "word_view" not in plain and viewed["word_view"]["storage_dtype"] == f"f{width}"
