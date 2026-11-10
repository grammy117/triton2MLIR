"""Constant f64 addition preserves arithmetic NaNs and the original old bits."""

import numpy as np
import pytest

from test_float_atomics import MERLIN, NATIVE, OPT, domain

pytestmark = pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin toolchains required")


@pytest.mark.parametrize("word", [0, 1 << 63, 1, 0x7ff0000000001234],
                         ids=["positive-zero", "negative-zero", "subnormal", "signaling-nan"])
def test_constant_f64_add_preserves_quiet_result_and_exact_old_words(tmp_path, word):
    from t2m.examples.atomic_kernels import constant_fadd
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    from test_integer_products import replay_library
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(constant_fadd, {"state": "*fp64", "old": "*fp64", "B": "constexpr", "WIDTH": "constexpr", "WORD": "constexpr"}, {"B": 16, "WIDTH": 64, "WORD": word}))
    abi = TensorABI("constant_fadd", (TensorBuffer("state", (16,), "readwrite", dtype="f64"),
        TensorBuffer("old", (16,), "write", dtype="f64")), grid=(1,))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile="guarded-buffer",
        address_policy="source-words", effect_profile="typed-atomics")
    incoming = domain(31, 64)[0][2][:16].copy()
    with np.errstate(invalid="ignore"):
        updated = np.add(incoming, np.uint64(word).view(np.float64))
    expected = [updated, incoming.copy()]
    replay_library(tmp_path, result, [([incoming], expected)])
    with np.load(tmp_path / "invoke0/evaluation/outputs.npz", allow_pickle=False) as outputs:
        actual, nan = outputs["out0"], np.isnan(updated)
        np.testing.assert_array_equal(np.isnan(actual), nan)
        np.testing.assert_array_equal(actual[~nan].view(np.uint64), updated[~nan].view(np.uint64))
        assert np.all(actual[nan].view(np.uint64) & np.uint64(1 << 51))
        np.testing.assert_array_equal(outputs["out1"].view(np.uint64), incoming.view(np.uint64))
