"""Original control positions, observed state, raw fill bits and byte custody."""

from copy import deepcopy
import json
import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
from t2m.word_evidence import check_word_stage, record, replay_events

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def capture(root, scalar=False, while_=False):
    from t2m.examples.word_event_kernels import conditional_events
    from t2m.frontend import capture_ttir
    path = root / "source.ttir"
    path.write_text(capture_ttir(conditional_events, {"x": "*fp32", "fill": "*fp32", "out": "*fp32", "n": "i32",
        "steps": "i32", "choose": "i1", "shift": "i32", "B": "constexpr", "SCALAR": "constexpr", "WHILE": "constexpr"},
        {"B": 8, "SCALAR": scalar, "WHILE": while_}))
    abi = TensorABI("conditional_events", (TensorBuffer("x", ("n",)), TensorBuffer("fill", ("n",)),
        TensorBuffer("out", ("n",), "readwrite")), grid=(GridExtent("n", 1 if scalar else 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)), RuntimeScalar("steps", "i32", 4, bounds=(-2, 3)),
                         RuntimeScalar("choose", "i1", 5), RuntimeScalar("shift", "i32", 6, bounds=(0, 128))))
    return convert_shared(path, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer", address_policy="source-words")


def cases():
    # Include negative zero, signed NaNs with payloads, infinities and normals.
    bits = np.array([0x80000000, 0x7fc12345, 0xffc54321, 0x7f800000, 0xff800000, 0x00000001, 0x3f800000], dtype=np.uint32)
    result = []
    for n, steps, choose, shift in ((0, 3, True, 128), (7, 0, True, 128), (9, 3, False, 128),
                                    (13, -2, True, 128), (1, 1, True, 0), (17, 2, True, 0), (31, 3, True, 0)):
        x = np.resize(bits, n).view(np.float32).copy()
        fill = np.resize(np.roll(bits, 2), n).view(np.float32).copy()
        old = np.arange(n, dtype=np.float32) - 99
        expected = np.where(np.arange(n) % 2 == 0, x, fill) if choose and steps > 0 else old.copy()
        arrays = [x, fill, old, np.asarray(n, np.int32), np.asarray(steps, np.int32),
                  np.asarray(choose, np.bool_), np.asarray(shift, np.int32)]
        result.append((arrays, expected))
    return result


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("while_", [False, True])
def test_region_events_preserve_raw_fill_bits_and_loop_observations(tmp_path, scalar, while_):
    from t2m.native import execute_native
    result = capture(tmp_path, scalar, while_)
    stage = json.loads(result.auxiliary_artifacts["word-stage.json"])
    check_word_stage(stage, result.auxiliary_artifacts["word-input.mlir"], result.auxiliary_artifacts["word-tensor.mlir"])
    replay_events(stage, result.auxiliary_artifacts["word-input.mlir"])
    assert result.trace["memory"]["word_conversion"] == "native-isolated-events-v2"
    assert "tt.call" not in stage["compute"]["input"]
    for i, (arrays, expected) in enumerate(cases()):
        copies = [a.copy() for a in arrays]
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                     shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        for value, copy in zip(arrays, copies, strict=True):
            assert value.tobytes() == copy.tobytes()


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("while_", [False, True])
def test_merlin_reuses_original_regions_and_memory_state(tmp_path, scalar, while_):
    from test_grid_domain_bounds import replay_library
    domains = cases()
    replay_library(tmp_path, capture(tmp_path, scalar, while_), domains)
    for i, (_, expected) in enumerate(domains):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as actual:
            np.testing.assert_array_equal(actual["out0"].view(np.uint32), expected.view(np.uint32))


def test_event_chain_and_typed_reconstruction_refuse_rehashed_changes(tmp_path):
    result = capture(tmp_path)
    source, output = (result.auxiliary_artifacts[n] for n in ("word-input.mlir", "word-tensor.mlir"))
    stage = json.loads(result.auxiliary_artifacts["word-stage.json"])
    for change in ("pointer-width", "parent-pipeline", "optimizer", "event-bytes", "compute-bytes", "binding", "lost-event"):
        bad = deepcopy(stage)
        events, compute = bad["events"], bad["compute"]
        if change == "pointer-width":
            command = events["native_stage"]["command"]
            command[command.index("--triton-to-unstructured=offset-bit-width=64")] = "--triton-to-unstructured=offset-bit-width=32"
        elif change == "parent-pipeline":
            compute["native_stage"]["command"].insert(1, "--triton-to-unstructured=offset-bit-width=64")
        elif change == "optimizer":
            compute["native_stage"]["optimizer"]["sha256"] = "f" * 64
        elif change in ("event-bytes", "compute-bytes"):
            selected = events if change == "event-bytes" else compute
            selected["input"] += "\n"
        elif change == "binding":
            events["bindings"][0]["input_types"][1] = "i32"
        else:
            events["bindings"].pop()
        with pytest.raises(ValueError):
            check_word_stage(bad, source, output)
            replay_events(bad, source)
    # A stage hash updated to match altered template bytes still fails complete
    # reconstruction from the original typed parent, rather than trusting JSON.
    bad = deepcopy(stage)
    bad["events"]["input"] += "\n"
    bad["events"]["native_stage"]["artifacts"]["source.ttir"] = record(bad["events"]["input"])
    check_word_stage(bad, source, output)
    with pytest.raises(ValueError, match="complete typed source event inventory"):
        replay_events(bad, source)
    from xdsl.dialects import builtin
    from xdsl.rewriter import Rewriter
    from t2m.shared_tensor import parse_shared
    from t2m.word_events import print_ir
    bad = deepcopy(stage)
    converted = parse_shared(bad["events"]["output"])
    argument = next(iter(converted.ops)).regions[0].block.args[1]
    Rewriter.replace_value_with_new_type(argument, builtin.TensorType(builtin.i32, argument.type.get_shape()))
    bad["events"]["output"] = print_ir(converted)
    bad["events"]["native_stage"]["artifacts"]["lowered.mlir"] = record(bad["events"]["output"])
    check_word_stage(bad, source, output)
    with pytest.raises(ValueError, match="typed capture bindings"):
        replay_events(bad, source)
