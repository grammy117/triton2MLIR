"""Equal-width logical storage views keep bits, fills, loops and masks."""

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


def capture(root, width, selected=False, before=False, while_=False, *, storage_width=None, scalar=False):
    from t2m.examples.pointer_view_kernels import word_view_memory
    from t2m.frontend import capture_ttir
    storage_width = storage_width or width
    constants = {"B": 8, "WIDTH": width, "SELECT": selected, "BEFORE": before, "WHILE": while_, "SCALAR": scalar}
    source = root / "source.ttir"
    source.write_text(capture_ttir(word_view_memory, {**{name: f"*fp{storage_width}" for name in ("x", "y", "out")},
        "n": "i32", "steps": "i32", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("word_view_memory", tuple(TensorBuffer(name, (4, "n"), "readwrite" if name == "out" else "read",
        dtype=f"f{storage_width}") for name in ("x", "y", "out")), grid=(GridExtent("n", 1 if scalar else 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)), RuntimeScalar("steps", "i32", 4, bounds=(0, 4))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words")


def cases(width, selected):
    uint, fp = getattr(np, f"uint{width}"), getattr(np, f"float{width}")
    max_ = (1 << width) - 1
    patterns = [0, 1, max_, (1 << (width - 1)), max_ >> 1, 0x1111, max_ - 16]
    for n, steps in ((0, 0), (1, 0), (13, 2), (31, 4)):
        x = np.resize(np.array(patterns, dtype=uint), (4, n))
        y = np.resize(np.array(patterns[::-1], dtype=uint), (4, n))
        initial = np.resize(np.array(patterns[1:] + patterns[:1], dtype=uint), (4, n))
        expected = initial.copy()
        lanes = np.arange(n)
        for row in range(steps):
            values = np.where(lanes % 2 == 0, x[row], y[row]) if selected else x[row]
            words = np.where(lanes % 3 != 1, values, uint((1 << width) - 17))
            expected[row] = words ^ uint(1 << (width - 1))
        arrays = [x.view(fp), y.view(fp), initial.view(fp), np.asarray(n, np.int32), np.asarray(steps, np.int32)]
        yield arrays, [expected.view(fp)]


@NATIVE_REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("selected,before,while_", [(False, False, False), (True, False, False), (True, True, True)])
def test_native_pointer_views_are_register_bitcasts_on_original_storage(tmp_path, width, selected, before, while_):
    from t2m.native import execute_native
    result = capture(tmp_path, width, selected, before, while_)
    reports = [r for r in result.trace["transformations"] if r["kind"] == "logical-pointer-word-view"]
    assert reports and {r["storage_dtype"] for r in reports} == {f"f{width}"}
    assert {r["view_dtype"] for r in reports} == {f"i{width}"}
    assert "arith.bitcast" in result.mlir_text and "ptr." not in result.mlir_text
    assert "prov.logical_word_views" in result.mlir_text
    for i, (arrays, expected) in enumerate(cases(width, selected)):
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
            shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual[0].view(getattr(np, f"uint{width}")), expected[0].view(getattr(np, f"uint{width}")))


@NATIVE_REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("width", [16, 32, 64])
def test_merlin_retains_word_views_across_dynamic_while_selection_and_zero_trips(tmp_path, width):
    from test_integer_products import replay_library
    result = capture(tmp_path, width, selected=True, before=True, while_=True)
    domains = list(cases(width, True))
    replay_library(tmp_path, result, domains)
    for i, (_, expected) in enumerate(domains):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
            np.testing.assert_array_equal(outputs["out0"].view(getattr(np, f"uint{width}")),
                expected[0].view(getattr(np, f"uint{width}")))


def test_pointer_type_parameters_require_equal_byte_aligned_words():
    from t2m.pointer_views import compatible, element
    from t2m.shared_tensor import parse_shared
    source = '''builtin.module { "tt.func"() <{sym_name="types", function_type=(!tt.ptr<f32>, !tt.ptr<i32>, !tt.ptr<i64>, !tt.ptr<i1>) -> ()}> ({
        ^bb0(%a: !tt.ptr<f32>, %b: !tt.ptr<i32>, %c: !tt.ptr<i64>, %d: !tt.ptr<i1>): "tt.return"() : () -> ()}) : () -> () }'''
    a, b, c, d = (v.type for v in next(iter(parse_shared(source).ops)).regions[0].block.args)
    assert compatible(a, b) and not compatible(a, c) and not compatible(b, d)
    assert element(a).bitwidth == 32
    with pytest.raises(ValueError, match="byte-aligned"):
        element(d)


@NATIVE_REQUIRED
@pytest.mark.parametrize("storage_width,view_width", [(16, 32), (32, 16), (32, 64)])
def test_changed_storage_stride_is_a_located_refusal(tmp_path, storage_width, view_width):
    with pytest.raises(ConversionError, match="equal byte-aligned") as error:
        capture(tmp_path, view_width, storage_width=storage_width)
    assert error.value.source_id and error.value.opcode


@NATIVE_REQUIRED
@pytest.mark.parametrize("operation", ["min", "max"])
def test_legacy_atomic_profile_does_not_admit_floating_word_views(tmp_path, operation):
    from test_float_atomics import capture as capture_float_atomic
    with pytest.raises(ConversionError, match="atomic word views") as error:
        capture_float_atomic(tmp_path, operation=operation, effect_profile="atomics")
    assert error.value.opcode == "tt.atomic_rmw" and error.value.source_id


@NATIVE_REQUIRED
@pytest.mark.parametrize("width", [16, 32, 64])
def test_scalar_word_views_preserve_fills_and_selected_while_state(tmp_path, width):
    from t2m.native import execute_native
    result = capture(tmp_path, width, selected=True, before=True, while_=True, scalar=True)
    for i, (arrays, expected) in enumerate(cases(width, True)):
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
            shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual[0].view(getattr(np, f"uint{width}")), expected[0].view(getattr(np, f"uint{width}")))


def test_pointer_bitcast_cannot_expose_a_physical_address_as_a_word():
    from t2m.pointer_control import normalize_pointer_control
    from t2m.shared_tensor import parse_shared, _source_ids
    module = parse_shared('''builtin.module { "tt.func"() <{sym_name="observe", function_type=(!tt.ptr<i32>) -> ()}> ({
      ^bb0(%p: !tt.ptr<i32>):
        %word = "tt.bitcast"(%p) : (!tt.ptr<i32>) -> i64 loc("ttir.op.observe")
        "tt.return"() : () -> ()
    }) : () -> () }''')
    with pytest.raises(ConversionError, match="observe or escape") as error:
        normalize_pointer_control(module, _source_ids)
    assert error.value.source_id == "ttir.op.observe" and error.value.opcode == "tt.bitcast"
