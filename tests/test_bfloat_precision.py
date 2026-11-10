"""BF16 storage words and arithmetic retain the logical tensor ABI."""

import os

import numpy as np
import pytest

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
REQUIRED = pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin required")


def capture(root, profile="tensor", mode="copy", scalar=False):
    from t2m.examples.bfloat_kernels import bfloat_copy, bfloat_convert
    from t2m.frontend import capture_ttir
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    kernel = bfloat_copy if mode == "copy" else bfloat_convert
    constants = {"B": 64, **({"SCALAR": scalar} if mode == "copy" else {"NARROW": mode == "narrow"})}
    first, second = ("f32", "bf16") if mode == "narrow" else ("bf16", "f32") if mode == "widen" else ("bf16", "bf16")
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, {"x": "*" + ("fp32" if first == "f32" else "bf16"),
        "out": "*" + ("fp32" if second == "f32" else "bf16"), "n": "i32", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", ("n",), dtype=first), TensorBuffer("out", ("n",), "write", dtype=second)),
        grid=(GridExtent("n", 1 if scalar else 64),), runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 65536)),))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")


@REQUIRED
def test_native_bfloat_copy_preserves_every_storage_word(tmp_path):
    from t2m.native import execute_native
    result = capture(tmp_path)
    words = np.arange(65536, dtype=np.uint16)
    (actual,), _ = execute_native(result.mlir_text, [words, np.asarray(words.size, np.int32)], tmp_path / "native", python=NATIVE,
        shape_abi=result.spec.runtime_shapes())
    np.testing.assert_array_equal(actual, words)


@REQUIRED
def test_native_bfloat_narrowing_has_executable_rounding(tmp_path):
    from t2m.native import execute_native
    result = capture(tmp_path, mode="narrow")
    values = np.array([1, 1 + 2**-8, 1 + 3 * 2**-8, -0.0, 2**-133, 2**-134, np.inf, np.nan], np.float32)
    (actual,), _ = execute_native(result.mlir_text, [values, np.asarray(values.size, np.int32)], tmp_path / "native", python=NATIVE,
        shape_abi=result.spec.runtime_shapes())
    np.testing.assert_array_equal(actual[:7], np.array([0x3f80, 0x3f80, 0x3f82, 0x8000, 1, 0, 0x7f80], np.uint16))
    assert actual[7] & 0x7f80 == 0x7f80 and actual[7] & 0x7f


def narrowing_reference(values):
    """Round binary32 words using Python quotient/remainder ties-to-even."""
    result = []
    for raw in values.view(np.uint32):
        word = int(raw)
        if word & 0x7fffffff > 0x7f800000:
            result.append(0x7fc0)
        else:
            quotient, remainder = divmod(word, 65536)
            result.append((quotient + (remainder > 32768 or remainder == 32768 and quotient % 2)) & 65535)
    return np.asarray(result, np.uint16)


def exact_numeric_words(actual, expected):
    # Computed NaN payload/sign are unspecified. All non-NaN words, including
    # signed zeros and subnormals, must match exactly.
    dtype = np.uint16 if actual.dtype == np.uint16 else np.uint32
    nan = (expected & 0x7f80 == 0x7f80) & (expected & 0x7f != 0) if dtype == np.uint16 else np.isnan(expected)
    np.testing.assert_array_equal(actual.view(dtype)[~nan], expected.view(dtype)[~nan])
    if dtype == np.uint16:
        assert np.all((actual[nan] & 0x7f80 == 0x7f80) & (actual[nan] & 0x7f != 0))
    else:
        assert np.all(np.isnan(actual[nan]))


@REQUIRED
@pytest.mark.parametrize("profile,scalar", [(p, False) for p in ("tensor", "buffer", "guarded-buffer")] + [("guarded-buffer", True)])
def test_bfloat_word_transport_through_reused_merlin_library(tmp_path, profile, scalar):
    from test_integer_products import replay_library
    result = capture(tmp_path, profile, scalar=scalar)
    words = np.arange(65536, dtype=np.uint16) if not scalar else np.array([0, 0x8000, 1, 0x7f80, 0x7f81, 0xffe5], np.uint16)
    cases = [([words[:n].copy(), np.asarray(n, np.int32)], [words[:n].copy()]) for n in (0, 1, 5, words.size)]
    replay_library(tmp_path, result, cases)
    for i, (_, expected) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as actual:
            np.testing.assert_array_equal(actual["out0"], expected[0])


@REQUIRED
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
@pytest.mark.parametrize("mode", ["narrow", "widen"])
def test_bfloat_casts_through_reused_merlin_library(tmp_path, profile, mode):
    from test_integer_products import replay_library
    result = capture(tmp_path, profile, mode=mode)
    assert "arith.truncf" in result.mlir_text if mode == "narrow" else "arith.extf" in result.mlir_text
    if mode == "narrow":
        # Both signs of all exponent fields; below/tie/above each boundary,
        # including minimum subnormals, finite overflow and signaling NaNs.
        words = [sign | exponent << 23 | mantissa for sign in (0, 0x80000000)
                 for exponent in range(256) for mantissa in (0, 1, 0x7fff, 0x8000, 0x8001, 0x18000, 0x7fffff)]
        values = np.asarray(words, np.uint32).view(np.float32)
        expected = narrowing_reference(values)
    else:
        values = np.arange(65536, dtype=np.uint16)
        expected = (values.astype(np.uint32) << 16).view(np.float32)
    cases = [([values[:n].copy(), np.asarray(n, np.int32)], [expected[:n].copy()]) for n in (0, 1, 65, values.size)]
    replay_library(tmp_path, result, cases)
    for i, (_, (reference,)) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as actual:
            exact_numeric_words(actual["out0"], reference)


def test_bfloat_comparison_decodes_numeric_values_and_requires_words():
    from t2m.tensor_storage import comparison_values
    values = comparison_values(np.array([0, 0x8000, 1, 0x3f80, 0xbf80, 0x7f81, 0x7fc0], np.uint16), "bf16")
    np.testing.assert_array_equal(values[:5], [0., -0., 2.**-133, 1., -1.])
    assert np.all(np.isnan(values[5:]))
    with pytest.raises(ValueError, match="raw uint16"):
        comparison_values(np.array([1.], np.float32), "bf16")


@REQUIRED
@pytest.mark.parametrize("profile,scalar", [(p, False) for p in ("tensor", "buffer", "guarded-buffer")] + [("guarded-buffer", True)])
def test_bfloat_runtime_select_loops_preserve_uncomputed_bits(tmp_path, profile, scalar):
    from test_integer_products import replay_library
    from t2m.examples.bfloat_kernels import bfloat_choose
    from t2m.frontend import capture_ttir
    from t2m.dynamic_abi import GridExtent
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(bfloat_choose, {"x": "*bf16", "y": "*bf16", "out": "*bf16", "n": "i32", "rounds": "i32",
        "B": "constexpr", "SCALAR": "constexpr"}, {"B": 64, "SCALAR": scalar}))
    abi = TensorABI("bfloat_choose", (TensorBuffer("x", ("n",), dtype="bf16"), TensorBuffer("y", ("n",), dtype="bf16"),
        TensorBuffer("out", ("n",), "write", dtype="bf16")), grid=(GridExtent("n", 1 if scalar else 64),),
        runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 65)), RuntimeScalar("rounds", "i32", 4, bounds=(0, 5))))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    words = np.array([0, 0x8000, 1, 0x807f, 0x7f81, 0xffa5, 0x7f80, 0xff80, 0x3f80], np.uint16)
    cases = []
    for n, rounds in ((0, 0), (1, 0), (17, 1), (65, 2), (65, 5)):
        x, y = np.resize(words, n), np.resize(words[::-1], n)
        expected = x.copy()
        for step in range(rounds):
            selected = (np.arange(n) + step) % 3 == 1
            expected[selected] = y[selected]
        cases.append(([x, y, np.asarray(n, np.int32), np.asarray(rounds, np.int32)], [expected]))
    replay_library(tmp_path, result, cases)
    for i, (_, (expected,)) in enumerate(cases):
        with np.load(tmp_path / f"invoke{i}/evaluation/outputs.npz", allow_pickle=False) as actual:
            np.testing.assert_array_equal(actual["out0"], expected)


@REQUIRED
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_bfloat_contraction_keeps_nonzero_float32_accumulator(tmp_path, profile):
    from test_half_precision import qualify
    from t2m.examples.tensor_kernels import accumulated_matmul
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(accumulated_matmul,
        {"x": "*bf16", "y": "*bf16", "initial": "*fp32", "out": "*fp32", "M": "constexpr", "N": "constexpr", "K": "constexpr"},
        {"M": 16, "N": 16, "K": 16}))
    abi = TensorABI("accumulated_matmul", (TensorBuffer("x", (16, 16), dtype="bf16"),
        TensorBuffer("y", (16, 16), dtype="bf16"), TensorBuffer("initial", (16, 16)), TensorBuffer("out", (16, 16), "write")))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured")
    matmul = next(op for op in result.module.walk() if op.name == "linalg.matmul")
    assert str(matmul.operands[0].type) == "tensor<16x16xbf16>"
    assert str(matmul.operands[2].type) == "tensor<16x16xf32>"
    rng = np.random.default_rng(71)
    x, y = (rng.integers(-2, 3, (16, 16)).astype(np.float32) for _ in range(2))
    initial = rng.integers(-5, 6, (16, 16)).astype(np.float32)
    # Exact small integers avoid assuming a particular backend reduction tree.
    words = [(value.view(np.uint32) >> 16).astype(np.uint16) for value in (x, y)]
    qualify(result, [*words, initial], initial + x @ y, tmp_path)
