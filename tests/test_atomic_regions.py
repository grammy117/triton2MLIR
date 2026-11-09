"""Atomic effects and returned values stay on original pointer/control edges."""

from copy import deepcopy
import json
import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared, opcode, parse_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def capture(root, width=32, scalar=False, while_=False, select="none"):
    from t2m.examples.atomic_kernels import region_atomic
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    constants = {"B": 8, "SCALAR": scalar, "WHILE": while_, "SELECT": select}
    source.write_text(capture_ttir(region_atomic, {"a": f"*i{width}", "b": f"*i{width}", "out": f"*i{width}",
        "n": "i32", "steps": "i32", "enabled": "i1", "choice": "i1", "shift": "i32",
        **{name: "constexpr" for name in constants}}, constants))
    dtype = f"i{width}"
    abi = TensorABI("region_atomic", (TensorBuffer("a", (4, "n"), "readwrite", dtype=dtype),
        TensorBuffer("b", (4, "n"), "readwrite", dtype=dtype), TensorBuffer("out", ("n",), "write", dtype=dtype)),
        grid=(GridExtent("n", 1 if scalar else 8),), runtime_scalars=(
            RuntimeScalar("n", "i32", 3, bounds=(0, 13)), RuntimeScalar("steps", "i32", 4, bounds=(0, 3)),
            RuntimeScalar("enabled", "i1", 5), RuntimeScalar("choice", "i1", 6),
            RuntimeScalar("shift", "i32", 7, bounds=(-1000, 1000))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT,
        memory_profile="guarded-buffer", address_policy="source-words", effect_profile="atomics")


def arguments(n, steps, enabled, choice, width=32, select="none", shift=0):
    dtype = getattr(np, f"int{width}")
    a = np.arange(4 * n, dtype=dtype).reshape(4, n) - 17
    b = 3 * np.arange(4 * n, dtype=dtype).reshape(4, n) + 50
    actual_a, actual_b, out = a.copy(), b.copy(), np.zeros(n, dtype)
    if enabled:
        for lane in range(n):
            target = actual_b if (select == "lane" and lane % 2 or select == "uniform" and choice) else actual_a
            total = 0
            for row in range(steps):
                previous = int(target[row, lane])
                target[row, lane] = previous ^ (row + 1) if row % 2 else previous + row + 1
                total += previous
            previous = int(target[steps, lane])
            target[steps, lane] = previous ^ total
            out[lane] = total + previous
    arrays = [a, b, np.asarray(n, np.int32), np.asarray(steps, np.int32), np.asarray(enabled, np.bool_),
              np.asarray(choice, np.bool_), np.asarray(shift, np.int32)]
    return arrays, [actual_a, actual_b, out]


def cases(width, select):
    return [arguments(n, steps, enabled, choice, width, select, shift) for n, steps, enabled, choice, shift in (
        (0, 3, True, False, 0), (1, 0, True, True, 0), (9, 2, True, False, 0),
        (13, 3, True, True, 0), (9, 3, False, True, 1000))]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar,while_,select", [(False, False, "none"), (True, True, "none"),
    (False, True, "uniform"), (True, False, "uniform"), (False, True, "lane"), (False, False, "lane")])
def test_regions_keep_atomic_recurrence_selected_bases_and_final_pointer(tmp_path, width, scalar, while_, select):
    from test_integer_products import replay_library, run_native
    result = capture(tmp_path, width, scalar, while_, select)
    stage = json.loads(result.auxiliary_artifacts["word-stage.json"])
    assert stage["schema"] == "t2m.source-word-stages.v2"
    assert any(item["opcode"] == "tt.atomic_rmw" for item in stage["events"]["bindings"])
    assert "scf.execute_region" in result.mlir_text and "no_inline" in result.mlir_text
    assert "scf.while" in result.mlir_text if while_ else "scf.for" in result.mlir_text
    domains = cases(width, select)
    run_native(tmp_path, result, domains)
    if MERLIN:
        replay_library(tmp_path, result, domains)


@pytest.mark.parametrize("shift", [-1, 1000])
def test_active_region_atomic_addresses_remain_guarded(tmp_path, shift):
    from t2m.native import execute_native
    result = capture(tmp_path, while_=True)
    arrays, _ = arguments(9, 3, False, False, shift=shift)
    arrays[4] = np.asarray(True, np.bool_)
    with pytest.raises(ValueError, match="exit -6"):
        execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=result.spec.runtime_shapes())


@pytest.mark.parametrize("attribute", ["atomic_rmw_op", "sem", "scope"])
def test_rehashed_event_output_cannot_change_atomic_semantics(tmp_path, attribute):
    from xdsl.dialects import builtin
    from t2m.diagnostics import ConversionError
    from t2m.word_events import print_ir
    from t2m.word_evidence import check_word_stage, record, replay_events
    result = capture(tmp_path)
    stage = deepcopy(json.loads(result.auxiliary_artifacts["word-stage.json"]))
    source, output = (result.auxiliary_artifacts[n] for n in ("word-input.mlir", "word-tensor.mlir"))
    events = parse_shared(stage["events"]["output"])
    memory = next(op for op in events.walk() if opcode(op) == "tts.unstructured_atomic_rmw")
    memory.attributes.pop(attribute, None)
    memory.properties[attribute] = builtin.IntegerAttr(2, builtin.i32)
    stage["events"]["output"] = print_ir(events)
    stage["events"]["native_stage"]["artifacts"]["lowered.mlir"] = record(stage["events"]["output"])
    check_word_stage(stage, source, output)
    with pytest.raises(ConversionError, match="source operation/order/scope"):
        replay_events(stage, source)
