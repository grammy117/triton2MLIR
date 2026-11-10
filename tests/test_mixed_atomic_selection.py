"""One source atomic may select integer storage or a floating word view."""

import os

import numpy as np
import pytest

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin required")


def cas_capture(root, width, scalar=False, lane=False, *, profile="typed-atomics"):
    from t2m.examples.atomic_kernels import selected_word_cas
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    source = root / "source.ttir"
    constants = {"B": 8, "WIDTH": width, "SCALAR": scalar, "LANE": lane}
    source.write_text(capture_ttir(selected_word_cas, {"a": f"*fp{width}", "b": f"*i{width}", "comparison": f"*i{width}",
        "value": f"*i{width}", "old": f"*fp{width}", "choice": "i1", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("selected_word_cas", (TensorBuffer("a", (16,), "readwrite", dtype=f"f{width}"),
        TensorBuffer("b", (16,), "readwrite", dtype=f"i{width}"), TensorBuffer("comparison", (16,), dtype=f"i{width}"),
        TensorBuffer("value", (16,), dtype=f"i{width}"), TensorBuffer("old", (16,), "write", dtype=f"f{width}")),
        grid=(16 if scalar else 2,), runtime_scalars=(RuntimeScalar("choice", "i1", 5),))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile=profile)


def cas_domains(width, lane):
    from test_atomic_word_cas import patterns
    uint, integer, floating = (getattr(np, f"{prefix}{width}") for prefix in ("uint", "int", "float"))
    for rotation, choice in ((0, False), (1, True), (4, False), (7, True)):
        a, b = np.resize(np.roll(patterns(width), rotation), 16), np.resize(np.roll(patterns(width), rotation + 3), 16)
        selected = np.arange(16) % 2 == 1 if lane else np.full(16, choice)
        old = np.where(selected, b, a)
        comparison = old.copy()
        comparison[1::3] ^= uint(1)
        comparison[:2] ^= uint(1 << (width - 1))
        desired = np.resize(np.roll(patterns(width), rotation + 7), 16)
        updated = np.where(old == comparison, desired, old)
        first, second = np.where(selected, a, updated), np.where(selected, updated, b)
        yield ([a.view(floating), b.view(integer), comparison.view(integer), desired.view(integer), np.asarray(choice, np.bool_)],
               [first.view(floating), second.view(integer), old.view(floating)])


def replay_words(root, result, domains, width):
    from test_atomic_word_views import exact
    from test_integer_products import replay_library
    replay_library(root, result, domains)
    for path, expected in [(root / "native/outputs.npz", domains[-1][1]),
            *[(root / f"invoke{i}/evaluation/outputs.npz", values) for i, (_, values) in enumerate(domains)]]:
        with np.load(path, allow_pickle=False) as actual:
            exact([actual[f"out{i}"] for i in range(3)], expected, width)
    payload = result.trace["memory"]["atomic_payload"]
    floating = [p for p in payload if p["buffer"] == "a"]
    integer = [p for p in payload if p["buffer"] == "b"]
    assert floating and integer and all(p.get("word_view") for p in floating)
    assert all("word_view" not in p for p in integer)
    assert any(set(a["source_ids"]) == set(b["source_ids"]) for a in floating for b in integer)


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("lane", [False, True])
def test_cas_keeps_root_storage_success_failure_and_original_old_words(tmp_path, width, scalar, lane):
    result = cas_capture(tmp_path, width, scalar, lane)
    replay_words(tmp_path, result, list(cas_domains(width, lane)), width)


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("profile", ["ordinary", "atomics"])
def test_selected_word_views_require_the_explicit_typed_effect_profile(tmp_path, width, profile):
    from t2m.diagnostics import ConversionError
    with pytest.raises(ConversionError, match="qualified|atomic word views") as error:
        cas_capture(tmp_path, width, lane=True, profile=profile)
    assert error.value.source_id and error.value.opcode


def rmw_capture(root, width, scalar=False, while_=False, lane=False, *, repeat=False):
    from t2m.examples.atomic_kernels import region_word_atomic, repeated_region_word_atomic
    from t2m.frontend import capture_ttir
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    kernel = repeated_region_word_atomic if repeat else region_word_atomic
    source = root / "source.ttir"
    constants = {"B": 8, "WIDTH": width, "SCALAR": scalar, "WHILE": while_, "SELECT": "lane" if lane else "uniform"}
    source.write_text(capture_ttir(kernel, {"a": f"*fp{width}", "b": f"*i{width}", "out": f"*fp{width}",
        "n": "i32", "steps": "i32", "enabled": "i1", "choice": "i1", "shift": "i32", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI(kernel.__name__, (TensorBuffer("a", (4, "n"), "readwrite", dtype=f"f{width}"),
        TensorBuffer("b", (4, "n"), "readwrite", dtype=f"i{width}"), TensorBuffer("out", (2, "n") if repeat else ("n",), "write", dtype=f"f{width}")),
        grid=(GridExtent("n", 1 if scalar else 8),), runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 13)),
        RuntimeScalar("steps", "i32", 4, bounds=(0, 3)), RuntimeScalar("enabled", "i1", 5), RuntimeScalar("choice", "i1", 6),
        RuntimeScalar("shift", "i32", 7, bounds=(-1000, 1000))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")


def rmw_domains(width, lane, *, repeat=False):
    from test_atomic_word_regions import cases
    integer = getattr(np, f"int{width}")
    for arrays, expected in cases(width, "lane" if lane else "uniform", repeat):
        arrays[1] = arrays[1].view(integer)
        expected[1] = expected[1].view(integer)
        yield arrays, expected


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("while_", [False, True])
@pytest.mark.parametrize("lane", [False, True])
def test_rmw_keeps_recurrences_masks_final_pointer_and_root_storage(tmp_path, width, scalar, while_, lane):
    result = rmw_capture(tmp_path, width, scalar, while_, lane)
    replay_words(tmp_path, result, list(rmw_domains(width, lane)), width)


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar,lane", [(False, False), (True, False), (False, True), (True, True)])
def test_repeated_selected_rmw_helpers_keep_each_root_and_call_context(tmp_path, width, scalar, lane):
    result = rmw_capture(tmp_path, width, scalar, not scalar, lane, repeat=True)
    replay_words(tmp_path, result, list(rmw_domains(width, lane, repeat=True)), width)
    payload = result.trace["memory"]["atomic_payload"]
    assert len({(p["buffer"], frozenset(p["source_ids"])) for p in payload}) == len(payload)


def primitive_capture(root, width, scalar, operation):
    from t2m.examples.atomic_kernels import selected_word_rmw
    from t2m.frontend import capture_ttir
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    source = root / "source.ttir"
    constants = {"B": 8, "WIDTH": width, "SCALAR": scalar, "OP": operation}
    source.write_text(capture_ttir(selected_word_rmw, {"a": f"*fp{width}", "b": f"*i{width}", "value": f"*i{width}",
        "index": "*i32", "old": f"*fp{width}", "n": "i32", "choice": "i1", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("selected_word_rmw", (TensorBuffer("a", (37,), "readwrite", dtype=f"f{width}"),
        TensorBuffer("b", (37,), "readwrite", dtype=f"i{width}"), TensorBuffer("value", ("n",), dtype=f"i{width}"),
        TensorBuffer("index", ("n",), dtype="i32"), TensorBuffer("old", ("n",), "write", dtype=f"f{width}")),
        grid=(GridExtent("n", 1 if scalar else 8),), runtime_scalars=(RuntimeScalar("n", "i32", 5, bounds=(0, 31)), RuntimeScalar("choice", "i1", 6)))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")


def primitive_domains(width, operation):
    from test_atomic_word_cas import patterns
    from test_atomic_memory import operation_reference, signed
    uint, integer, floating = (getattr(np, f"{prefix}{width}") for prefix in ("uint", "int", "float"))
    for n, choice in ((0, False), (1, True), (13, False), (31, True)):
        a, b = np.resize(patterns(width), 37), np.resize(np.roll(patterns(width), 5), 37)
        values = np.resize(np.roll(patterns(width), n % 12), n)
        index = (np.arange(n, dtype=np.int32) * 11 + 3) % 37
        active = np.arange(n) % 3 != 1
        index[~active] = -1000
        first, second, old = a.copy(), b.copy(), np.zeros(n, uint)
        for lane in range(n):
            if active[lane]:
                target = first if (lane % 2 == 0) == choice else second
                old[lane] = target[index[lane]]
                target[index[lane]] = uint(operation_reference(signed(int(old[lane]), width), signed(int(values[lane]), width), operation, width) & ((1 << width) - 1))
        yield ([a.view(floating), b.view(integer), values.view(integer), index, np.asarray(n, np.int32), np.asarray(choice, np.bool_)],
               [first.view(floating), second.view(integer), old.view(floating)])


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("operation", ["add", "and", "or", "xor", "max", "min", "umax", "umin", "exch"])
def test_all_integer_rmw_primitives_keep_selected_storage_and_loaded_indices(tmp_path, width, scalar, operation):
    result = primitive_capture(tmp_path, width, scalar, operation)
    replay_words(tmp_path, result, list(primitive_domains(width, operation)), width)


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_colliding_selected_roots_keep_independent_complete_word_histories(tmp_path, width, scalar):
    from t2m.bundle import _record
    from t2m.consumer import invoke_merlin_host
    from t2m.kernel_consumer import verify_kernel_qualification
    from t2m.native import execute_native
    from test_integer_products import replay_library
    result = primitive_capture(tmp_path, width, scalar, "add")
    baseline = list(primitive_domains(width, "add"))[-1:]
    replay_library(tmp_path, result, baseline)
    qualification = tmp_path / "merlin"
    receipt, library = verify_kernel_qualification(qualification), _record(qualification / "host/model.so")
    uint, integer, floating = (getattr(np, f"{prefix}{width}") for prefix in ("uint", "int", "float"))
    mask = (1 << width) - 1
    for case, (n, choice) in enumerate(((1, False), (13, True), (31, False))):
        arrays, _ = list(primitive_domains(width, "add"))[-1]
        a, b = arrays[0].copy().view(uint), arrays[1].copy().view(uint)
        a[0] = uint(0x7f800001 if width == 32 else 0x7ff0000000000001)
        b[0] = uint(mask - 2)
        active = np.arange(n) % 3 != 1
        indices = np.zeros(n, np.int32)
        indices[~active] = -1000
        inputs = [a.view(floating), b.view(integer), np.ones(n, integer), indices, np.asarray(n, np.int32), np.asarray(choice, np.bool_)]
        native, _ = execute_native(result.mlir_text, inputs, tmp_path / f"collision-native{case}", python=NATIVE,
            shape_abi=result.spec.runtime_shapes())
        arguments = tmp_path / f"collision-inputs{case}.npz"
        np.savez(arguments, **{f"in{i}": value for i, value in enumerate(inputs)})
        compiled, evaluation = invoke_merlin_host(qualification, arguments, tmp_path / f"collision-merlin{case}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert evaluation["functional_inputs_preserved"]
        take_a = (np.arange(n) % 2 == 0) == choice
        for first, second, old in (native, compiled):
            old = old.view(uint)
            np.testing.assert_array_equal(old[~active], np.zeros(np.count_nonzero(~active), uint))
            for actual, incoming, selected in ((first, a, take_a), (second, b, ~take_a)):
                selected = selected & active
                count = int(np.count_nonzero(selected))
                expected = incoming.copy()
                expected[0] = uint((int(incoming[0]) + count) & mask)
                np.testing.assert_array_equal(actual.view(uint), expected)
                history = np.asarray([(int(incoming[0]) + i) & mask for i in range(count)], uint)
                np.testing.assert_array_equal(np.sort(old[selected]), np.sort(history))
        assert verify_kernel_qualification(qualification) == receipt
        assert _record(qualification / "host/model.so") == library
