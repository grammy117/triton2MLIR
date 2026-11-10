"""One original atomic helper may act on integer storage and floating word storage."""

import os

import numpy as np
import pytest

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin required")


@pytest.mark.parametrize("width", [32, 64])
def test_one_helper_retains_distinct_integer_and_float_storage_contexts(tmp_path, width):
    from t2m.examples.atomic_kernels import mixed_storage_atomic
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from test_float_atomics import domain
    from test_integer_products import replay_library
    integer, floating, uint = getattr(np, f"int{width}"), getattr(np, f"float{width}"), getattr(np, f"uint{width}")
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(mixed_storage_atomic, {"integer_state": f"*i{width}", "floating_state": f"*fp{width}",
        "old_integer": f"*i{width}", "old_word": f"*fp{width}", "B": "constexpr", "WIDTH": "constexpr"}, {"B": 8, "WIDTH": width}))
    abi = TensorABI("mixed_storage_atomic", (TensorBuffer("integer_state", (16,), "readwrite", dtype=f"i{width}"),
        TensorBuffer("floating_state", (16,), "readwrite", dtype=f"f{width}"),
        TensorBuffer("old_integer", (16,), "write", dtype=f"i{width}"), TensorBuffer("old_word", (16,), "write", dtype=f"f{width}")), grid=(2,))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    incoming_integer = np.arange(16, dtype=integer) - 19
    incoming_word = domain(31, width)[0][2][:16].copy()
    mask = (1 << width) - 1
    updated = np.array([(int(word) + lane + 1) & mask for lane, word in enumerate(incoming_word.view(uint))], uint).view(floating)
    expected = [incoming_integer + np.arange(1, 17, dtype=integer), updated, incoming_integer.copy(), incoming_word.copy()]
    replay_library(tmp_path, result, [([incoming_integer, incoming_word], expected)])
    with np.load(tmp_path / "invoke0/evaluation/outputs.npz", allow_pickle=False) as outputs:
        for i, reference in enumerate(expected):
            np.testing.assert_array_equal(outputs[f"out{i}"].view(uint), reference.view(uint))
    payload = result.trace["memory"]["atomic_payload"]
    plain = next(event for event in payload if event["buffer"] == "integer_state")
    viewed = next(event for event in payload if event["buffer"] == "floating_state")
    assert set(plain["source_ids"]) & set(viewed["source_ids"])  # same original callee operation
    assert set(plain["source_ids"]) != set(viewed["source_ids"])  # different call contexts
    assert "word_view" not in plain and viewed["word_view"]["storage_dtype"] == f"f{width}"
