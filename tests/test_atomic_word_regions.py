"""Word atomic recurrence and selection preserve raw floating storage bits."""

import json
import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and os.environ.get("T2M_MERLIN_ROOT")), reason="existing shared/native/explicit compiler support required")


def capture(root, width=32, scalar=False, while_=False, select="none", *, repeat=False):
    from t2m.examples.atomic_kernels import region_word_atomic, repeated_region_word_atomic
    kernel = repeated_region_word_atomic if repeat else region_word_atomic
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    constants = {"B": 8, "WIDTH": width, "SCALAR": scalar, "WHILE": while_, "SELECT": select}
    source.write_text(capture_ttir(kernel, {"a": f"*fp{width}", "b": f"*fp{width}", "out": f"*fp{width}",
        "n": "i32", "steps": "i32", "enabled": "i1", "choice": "i1", "shift": "i32",
        **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI(kernel.__name__, (TensorBuffer("a", (4, "n"), "readwrite", dtype=f"f{width}"),
        TensorBuffer("b", (4, "n"), "readwrite", dtype=f"f{width}"), TensorBuffer("out", (2, "n") if repeat else ("n",), "write", dtype=f"f{width}")),
        grid=(GridExtent("n", 1 if scalar else 8),), runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 13)),
        RuntimeScalar("steps", "i32", 4, bounds=(0, 3)), RuntimeScalar("enabled", "i1", 5), RuntimeScalar("choice", "i1", 6),
        RuntimeScalar("shift", "i32", 7, bounds=(-1000, 1000))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")


def apply(a, b, width, select, n, steps, enabled, choice):
    from test_atomic_memory import operation_reference, signed
    uint, mask = getattr(np, f"uint{width}"), (1 << width) - 1
    first, second, out = a.copy(), b.copy(), np.zeros(n, dtype=uint)
    if enabled:
        for lane in range(n):
            target = second if select == "lane" and lane % 2 or select == "uniform" and choice else first
            total = 0
            for row in range(steps):
                previous = int(target[row, lane])
                target[row, lane] = uint(operation_reference(signed(previous, width), row + 1,
                    "xor" if row % 2 else "add", width) & mask)
                total = (total + previous) & mask
            previous = int(target[steps, lane])
            target[steps, lane] = uint(previous ^ total)
            out[lane] = uint((total + previous) & mask)
    return first, second, out


def cases(width, select, repeat=False):
    from test_float_atomics import domain
    uint, fp = getattr(np, f"uint{width}"), getattr(np, f"float{width}")
    patterns = domain(31, width)[0][0].view(uint)
    for n, steps, enabled, choice, shift in ((0, 3, True, False, 0), (1, 0, True, True, 0),
            (9, 2, True, False, 0), (13, 3, True, True, 0), (9, 3, False, True, 1000)):
        a, b = np.resize(patterns, (4, n)), np.resize(patterns[::-1], (4, n))
        first, second, out = apply(a, b, width, select, n, steps, enabled, choice)
        if repeat:
            first, second, other = apply(first, second, width, select, n, steps, enabled, not choice)
            out = np.stack([out, other])
        arrays = [a.view(fp), b.view(fp), np.asarray(n, np.int32), np.asarray(steps, np.int32),
            np.asarray(enabled, np.bool_), np.asarray(choice, np.bool_), np.asarray(shift, np.int32)]
        yield arrays, [first.view(fp), second.view(fp), out.view(fp)]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar,while_,select", [(False, False, "none"), (True, True, "none"),
    (False, True, "uniform"), (True, False, "uniform"), (False, True, "lane"), (False, False, "lane")])
def test_word_regions_keep_recurrence_selection_and_final_pointer(tmp_path, width, scalar, while_, select):
    from test_atomic_word_views import exact
    from test_integer_products import replay_library
    from t2m.native import execute_native
    result = capture(tmp_path, width, scalar, while_, select)
    domains = list(cases(width, select))
    for i, (arrays, expected) in enumerate(domains):
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
            shape_abi=result.spec.runtime_shapes())
        exact(actual, expected, width)
    if MERLIN:
        replay_library(tmp_path, result, domains)
        for i, (_, expected) in enumerate(domains):
            with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
                exact([outputs[f"out{j}"] for j in range(3)], expected, width)
    assert all(event.get("word_view") for event in result.trace["memory"]["atomic_payload"])
    assert json.loads(result.auxiliary_artifacts["word-stage.json"])["schema"] == "t2m.source-word-stages.v2"


@pytest.mark.parametrize("width", [32, 64])
def test_repeated_inlined_helpers_keep_separate_memory_contexts(tmp_path, width):
    from test_atomic_word_views import exact
    from test_integer_products import replay_library
    result = capture(tmp_path, width, while_=True, select="uniform", repeat=True)
    domains = list(cases(width, "uniform", repeat=True))
    payload = result.trace["memory"]["atomic_payload"]
    assert len({(event["buffer"], frozenset(event["source_ids"])) for event in payload}) == len(payload)
    if MERLIN:
        replay_library(tmp_path, result, domains)
        for i, (_, expected) in enumerate(domains):
            with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as outputs:
                exact([outputs[f"out{j}"] for j in range(3)], expected, width)
