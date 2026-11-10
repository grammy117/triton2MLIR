"""Finite selectors retain each buffer's actual storage and logical word values."""

import os

import numpy as np
import pytest

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
REQUIRED = pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin required")


def capture(root, width, scalar=False, while_=True, storage_width=None):
    from t2m.examples.pointer_view_kernels import word_view_memory
    from t2m.frontend import capture_ttir
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    constants = {"B": 8, "WIDTH": width, "SELECT": True, "BEFORE": True, "WHILE": while_, "SCALAR": scalar}
    storage_width = storage_width or width
    source = root / "source.ttir"
    source.write_text(capture_ttir(word_view_memory, {"x": f"*i{width}", "y": f"*fp{storage_width}", "out": f"*fp{width}",
        "n": "i32", "steps": "i32", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("word_view_memory", (TensorBuffer("x", (4, "n"), dtype=f"i{width}"),
        TensorBuffer("y", (4, "n"), dtype=f"f{storage_width}"), TensorBuffer("out", (4, "n"), "readwrite", dtype=f"f{width}")),
        grid=(GridExtent("n", 1 if scalar else 8),), runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)),
        RuntimeScalar("steps", "i32", 4, bounds=(0, 4))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer", address_policy="source-words")


@REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("scalar,while_", [(False, True), (True, False), (False, False), (True, True)])
def test_lane_selection_keeps_integer_and_float_storage_across_pointer_recurrences(tmp_path, width, scalar, while_):
    from test_pointer_views import cases
    from test_integer_products import replay_library
    from test_atomic_word_views import exact
    result = capture(tmp_path, width, scalar, while_)
    domains = []
    for arrays, expected in cases(width, True):
        arrays[0] = arrays[0].view(getattr(np, f"int{width}"))
        domains.append((arrays, expected))
    replay_library(tmp_path, result, domains)
    for i, (_, expected) in enumerate(domains):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as actual:
            exact([actual["out0"]], expected, width)
    reports = [r for r in result.trace["transformations"] if r["kind"] == "logical-pointer-word-view"]
    assert any(r["roots"] == [1] and r["storage_dtype"] == f"f{width}" for r in reports)
    assert "prov.logical_word_views" in result.mlir_text and "llvm." not in result.mlir_text


def update_capture(root, width, scalar, while_, lane):
    from t2m.examples.pointer_view_kernels import mixed_storage_updates
    from t2m.frontend import capture_ttir
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    constants = {"B": 8, "WIDTH": width, "SCALAR": scalar, "WHILE": while_, "LANE": lane}
    source = root / "source.ttir"
    source.write_text(capture_ttir(mixed_storage_updates, {"x": f"*i{width}", **{name: f"*fp{width}" for name in ("y", "before", "after", "final")},
        "n": "i32", "steps": "i32", "choice": "i1", "enabled": "i1", "shift": "i32", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("mixed_storage_updates", (TensorBuffer("x", (4, "n"), "readwrite", dtype=f"i{width}"),
        *(TensorBuffer(name, (4, "n"), "readwrite", dtype=f"f{width}") for name in ("y", "before", "after")),
        TensorBuffer("final", ("n",), "write", dtype=f"f{width}")), grid=(GridExtent("n", 1 if scalar else 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 5, bounds=(0, 31)), RuntimeScalar("steps", "i32", 6, bounds=(0, 4)),
            RuntimeScalar("choice", "i1", 7), RuntimeScalar("enabled", "i1", 8), RuntimeScalar("shift", "i32", 9, bounds=(-1000, 1000))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer", address_policy="source-words")


def update_domains(width, lane):
    from test_atomic_word_cas import patterns
    uint, integer, floating = (getattr(np, f"{prefix}{width}") for prefix in ("uint", "int", "float"))
    mask = (1 << width) - 1
    for n, steps, choice, enabled, shift in ((0, 4, False, True, 0), (1, 0, True, True, 0),
            (13, 2, True, True, 0), (17, 3, False, True, 0), (7, 3, True, False, 1000),
            (9, 0, True, True, 0 if lane else 1000), (31, 4, True, True, 0)):
        incoming = [np.resize(np.roll(patterns(width), rotation), (4, n)) for rotation in (0, 3, 5, 7)]
        x, y, before, after = [value.copy() for value in incoming]
        selected = (np.arange(n) % 2 == 0) == choice if lane else np.full(n, choice)
        fill = np.asarray([(i * 7 - 17) & mask for i in range(n)], uint)
        for row in range(steps if enabled else 0):
            take_x = selected if row % 2 == 0 else ~selected
            active = np.arange(n) % 3 != 1
            current = np.where(take_x, x[row], y[row])
            before[row] = np.where(active, current, fill)
            updated = current ^ uint(1 << (width - 1))
            x[row] = np.where(take_x & active, updated, x[row])
            y[row] = np.where(~take_x & active, updated, y[row])
            after[row] = np.where(active, updated, fill)
        if enabled and steps < 4:
            take_x = selected if steps % 2 == 0 else ~selected
            final = np.where(take_x, x[steps], y[steps])
        else:
            final = np.full(n, uint(mask - 16), uint)
        arrays = [incoming[0].view(integer), *(value.view(floating) for value in incoming[1:]),
            np.asarray(n, np.int32), np.asarray(steps, np.int32), np.asarray(choice, np.bool_), np.asarray(enabled, np.bool_), np.asarray(shift, np.int32)]
        expected = [x.view(integer), *(value.view(floating) for value in (y, before, after, final))]
        yield arrays, expected


@REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("scalar,while_,lane", [(False, True, False), (True, False, True), (False, False, True), (True, True, False),
    (False, False, False), (False, True, True), (True, False, False), (True, True, True)])
def test_selected_updates_keep_effect_order_backedges_and_final_pointer_storage(tmp_path, width, scalar, while_, lane):
    from test_integer_products import replay_library
    from test_atomic_word_views import exact
    result = update_capture(tmp_path, width, scalar, while_, lane)
    if not scalar:
        assert any(r["kind"] == "logical-selector-permutation" for r in result.trace["transformations"])
    domains = list(update_domains(width, lane))
    replay_library(tmp_path, result, domains)
    compatibility = (tmp_path / "capture/compatibility.json").read_text()
    assert "each root's own equal-width storage dtype" in compatibility
    if not scalar:
        assert "scalar i32 table indices" in compatibility
    for path, expected in [(tmp_path / "native/outputs.npz", domains[-1][1]),
            *[(tmp_path / f"invoke{i}/evaluation/outputs.npz", values) for i, (_, values) in enumerate(domains)]]:
        with np.load(path, allow_pickle=False) as actual:
            exact([actual[f"out{i}"] for i in range(5)], expected, width)


@REQUIRED
@pytest.mark.parametrize("width,storage_width", [(16, 32), (32, 16), (64, 32)])
def test_mixed_roots_still_refuse_changed_element_address_units(tmp_path, width, storage_width):
    from t2m.diagnostics import ConversionError
    with pytest.raises(ConversionError, match="equal byte-aligned") as error:
        capture(tmp_path, width, storage_width=storage_width)
    assert error.value.source_id and error.value.opcode


@REQUIRED
def test_each_selected_root_retains_active_bounds_checks(tmp_path):
    from t2m.native import execute_native
    result = update_capture(tmp_path, 32, False, True, False)
    arrays, _ = list(update_domains(32, False))[5]  # Dormant float pointer has shift 1000.
    arrays[6] = np.asarray(False, np.bool_)  # Activate the out-of-bounds float root.
    with pytest.raises(ValueError, match="exit -6"):
        execute_native(result.mlir_text, arrays, tmp_path / "invalid", python=NATIVE, shape_abi=result.spec.runtime_shapes())


@REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("while_", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_three_way_selector_cycles_keep_every_initial_tile_and_phase(tmp_path, width, while_, reverse):
    from t2m.examples.pointer_view_kernels import three_storage_permutation
    from t2m.frontend import capture_ttir
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from test_atomic_word_cas import patterns
    from test_atomic_word_views import exact
    from test_integer_products import replay_library
    source = tmp_path / "source.ttir"
    constants = {"B": 8, "WIDTH": width, "WHILE": while_, "REVERSE": reverse}
    source.write_text(capture_ttir(three_storage_permutation, {"x": f"*i{width}", "y": f"*fp{width}", "z": f"*i{width}", "out": f"*i{width}",
        "n": "i32", "steps": "i32", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("three_storage_permutation", (TensorBuffer("x", (4, "n"), dtype=f"i{width}"),
        TensorBuffer("y", (4, "n"), dtype=f"f{width}"), TensorBuffer("z", (4, "n"), dtype=f"i{width}"),
        TensorBuffer("out", (4, "n"), "readwrite", dtype=f"i{width}")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)), RuntimeScalar("steps", "i32", 5, bounds=(0, 3))))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer", address_policy="source-words")
    reports = [r for r in result.trace["transformations"] if r["kind"] == "logical-selector-permutation"]
    assert len(reports) == 1 and len(reports[0]["state_positions"]) == 3
    integer, floating = getattr(np, f"int{width}"), getattr(np, f"float{width}")
    domains = []
    for n, steps in ((0, 3), (1, 0), (13, 1), (17, 2), (31, 3)):
        values = [np.resize(np.roll(patterns(width), rotation), (4, n)) for rotation in (0, 3, 5, 7)]
        expected = values[3].copy()
        lanes = np.arange(n)
        roots = [np.where(lanes % 2 == 0, 0, 1), np.where(lanes % 3 == 0, 1, 2), np.where(lanes % 5 == 0, 2, 0)]
        for row in range(steps + 1):
            selected = roots[(-row if reverse else row) % 3]
            expected[row] = np.choose(selected, [v[row] for v in values[:3]])
        domains.append(([values[0].view(integer), values[1].view(floating), values[2].view(integer), values[3].view(integer),
            np.asarray(n, np.int32), np.asarray(steps, np.int32)], [expected.view(integer)]))
    replay_library(tmp_path, result, domains)
    for i, (_, expected) in enumerate(domains):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as actual:
            exact([actual["out0"]], expected, width)
