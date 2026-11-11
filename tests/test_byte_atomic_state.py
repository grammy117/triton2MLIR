"""Selected byte words, narrow updates and source pointer recurrences."""

import os

import pytest

from t2m.byte_atomic_state import convert
from t2m.examples.byte_atomic_state_reference import cases

OPT = os.environ.get("TRITON_SHARED_OPT")
NATIVE = os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing pinned shared/native toolchains required")


@pytest.mark.parametrize("width,cas", ((32, False), (64, False), (16, True), (32, True), (64, True)))
@pytest.mark.parametrize("while_", (False, True))
@pytest.mark.parametrize("order", ("little", "big"))
def test_selected_word_updates_keep_original_byte_state_and_recurrence(tmp_path, width, cas, while_, order):
    from t2m.native import execute_native

    result = convert(tmp_path / "source", width=width, cas=cas, while_=while_, byte_order=order, optimizer=OPT)
    assert any(t["kind"] == "selected-pointer-control" for t in result.trace["transformations"])
    assert f"scf.{'while' if while_ else 'for'}" in result.mlir_text
    for i, (arrays, expected) in enumerate(cases(width, cas, byte_order=order)):
        observed, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                     shape_abi=result.spec.runtime_shapes(), integer_atomic_regions=True)
        for actual, reference in zip(observed, expected, strict=True):
            assert actual.tobytes() == reference.tobytes()


@pytest.mark.skipif(not MERLIN, reason="existing explicit Merlin toolchain required")
@pytest.mark.parametrize("width,cas", ((32, False), (64, False), (16, True), (32, True), (64, True)))
@pytest.mark.parametrize("while_", (False, True))
@pytest.mark.parametrize("order", ("little", "big"))
def test_merlin_libraries_retain_selected_roots_loops_and_narrow_updates(tmp_path, width, cas, while_, order):
    from test_integer_products import replay_library

    result = convert(tmp_path / "source", width=width, cas=cas, while_=while_, byte_order=order, optimizer=OPT)
    replay_library(tmp_path, result, cases(width, cas, byte_order=order))


@pytest.mark.skipif(not MERLIN, reason="existing explicit Merlin toolchain required")
@pytest.mark.parametrize("width,cas", ((32, False), (64, False), (16, True), (32, True), (64, True)))
@pytest.mark.parametrize("while_", (False, True))
@pytest.mark.parametrize("order", ("little", "big"))
def test_original_atomic_old_value_tensor_recurrence_reaches_merlin(tmp_path, width, cas, while_, order):
    from test_integer_products import replay_library

    result = convert(tmp_path / "source", width=width, cas=cas, while_=while_, byte_order=order, optimizer=OPT, carry=True)
    replay_library(tmp_path, result, cases(width, cas, byte_order=order, carry=True))


@pytest.mark.skipif(not MERLIN, reason="existing explicit Merlin toolchain required")
@pytest.mark.parametrize("width,cas,root_dtype,while_", (
    (32, False, "i64", True), (64, False, "f16", True),
    (16, True, "f64", False), (64, True, "i32", False)))
@pytest.mark.parametrize("order", ("little", "big"))
def test_atomic_old_values_recur_over_different_width_typed_roots(tmp_path, width, cas, root_dtype, while_, order):
    from test_integer_products import replay_library

    result = convert(tmp_path / "source", width=width, cas=cas, while_=while_, byte_order=order,
                     root_dtype=root_dtype, carry=True, optimizer=OPT)
    replay_library(tmp_path, result, cases(width, cas, byte_order=order, root_dtype=root_dtype, carry=True))
