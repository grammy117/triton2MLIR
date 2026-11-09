"""Selected reads retain word arithmetic, defined fills and raw value bits."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
NP_TYPES = {"f16": np.float16, "f32": np.float32, "f64": np.float64, "i32": np.int32}


def capture(root, *, dtype="f32", words=False, cancel=False, omit=False):
    from t2m.examples.selected_pointer_kernels import selected_fills, selected_words
    from t2m.frontend import capture_ttir
    kernel = selected_words if words else selected_fills
    third = "encoded" if words else "fills"
    source_dtype = {"f16": "fp16", "f32": "fp32", "f64": "fp64", "i32": "i32"}[dtype]
    signature = {"x": "*" + source_dtype, "y": "*" + source_dtype,
        third: "*i32" if words else "*" + source_dtype, "out": "*" + source_dtype,
        "n": "i32", "m": "i32", "choice": "i32", "B": "constexpr",
        "CANCEL" if words else "OMIT_OTHER": "constexpr"}
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, signature, {"B": 8, "CANCEL" if words else "OMIT_OTHER": cancel if words else omit}))
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", ("m",) if words else ("n",), dtype=dtype),
        TensorBuffer("y", ("m",), dtype=dtype), TensorBuffer(third, ("n",), dtype="i32" if words else dtype),
        TensorBuffer("out", ("n",), "readwrite" if omit else "write", dtype=dtype)), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)), RuntimeScalar("m", "i32", 5, bounds=(0, 127)),
                         RuntimeScalar("choice", "i32", 6, bounds=(0, 3))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT,
                          memory_profile="guarded-buffer", address_policy="source-words")


def bit_values(size, dtype, shift=0):
    patterns = {
        "f16": (np.uint16, [0x8000, 0, 0x7c00, 0xfc00, 0x7e31, 0x3c00]),
        "f32": (np.uint32, [0x80000000, 0, 0x7f800000, 0xff800000, 0x7fc12345, 0x3f800000]),
        "f64": (np.uint64, [0x8000000000000000, 0, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000001234, 0x3ff0000000000000]),
    }
    if dtype == "i32":
        return np.arange(size, dtype=np.int32) + 100 * shift
    unsigned, pattern = patterns[dtype]
    return np.resize(np.roll(np.asarray(pattern, dtype=unsigned), shift), size).view(NP_TYPES[dtype]).copy()


def arguments(n, choice=0, *, dtype="f32", words=False, cancel=False, omit=False, base=1 << 30):
    m = n + 2 if not words else n if cancel else 4 * n + 3
    if words:
        x, y = np.arange(m, dtype=np.float32) - 5, np.arange(m, dtype=np.float32) + 100
        third = (base + np.arange(n)).astype(np.int32)
        index = np.arange(n) if cancel else (third.astype(np.int64) * 4 + 1).astype(np.int32).astype(np.int64)
        expected = x[index].copy()
    else:
        x, y, third = (bit_values(size, dtype, shift) for size, shift in ((n, 0), (m, 1), (n, 2)))
        index = np.arange(n)
        expected = x.copy()
    take_x = (np.arange(n) + choice) % 2 == 0
    expected[~take_x] = y[m - 1 - index[~take_x]]
    old = np.full(n, -999, dtype=NP_TYPES[dtype])
    if not words:
        inactive = np.arange(n) % 3 == 1
        expected[inactive] = old[inactive] if omit else third[inactive]
    arrays = [x, y, third, *([old] if omit else []), *[np.asarray(v, dtype=np.int32) for v in (n, m, choice)]]
    return arrays, expected


def assert_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(f"u{actual.dtype.itemsize}"), expected.view(f"u{expected.dtype.itemsize}"))


@pytest.mark.parametrize("dtype", ["f16", "f32", "f64", "i32"])
def test_selected_reads_keep_nonuniform_fills_and_value_bits(tmp_path, dtype):
    from t2m.native import execute_native
    result = capture(tmp_path, dtype=dtype)
    for i, n in enumerate((0, 1, 13, 31)):
        arrays, expected = arguments(n, i % 2, dtype=dtype)
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                     shape_abi=result.spec.runtime_shapes())
        assert_bits(actual, expected)


def test_missing_other_refinement_keeps_source_and_inactive_output_state(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.shared_bundle import write_shared_bundle
    result = capture(tmp_path, omit=True)
    arrays, expected = arguments(13, 1, omit=True)
    out = tmp_path / "capture"
    write_shared_bundle(result, out, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    verify_bundle(out)
    assert "inactive undefined source load values" in (out / "compatibility.json").read_text()
    assert any(t.get("missing_other_zero_refinement") for t in result.trace["transformations"])


@pytest.mark.parametrize("cancel", [False, True])
def test_selected_addresses_preserve_word_wrap_and_pointer_cancellation(tmp_path, cancel):
    from t2m.native import execute_native
    result = capture(tmp_path, words=True, cancel=cancel)
    for i, (n, base) in enumerate(((0, 0), (1, 1 << 30), (13, 1 << 31), (31, 3 << 30))):
        arrays, expected = arguments(n, i % 2, words=True, cancel=cancel, base=base)
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                     shape_abi=result.spec.runtime_shapes())
        assert_bits(actual, expected)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("words,cancel,dtype", [(False, False, "f16"), (False, False, "f32"),
    (False, False, "f64"), (False, False, "i32"), (True, False, "f32"), (True, True, "f32")])
def test_one_merlin_library_retains_selected_values_and_encoded_words(tmp_path, words, cancel, dtype):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result = capture(tmp_path, dtype=dtype, words=words, cancel=cancel)
    arrays, expected = arguments(13, 1, dtype=dtype, words=words, cancel=cancel)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, (n, base) in enumerate(((0, 0), (1, 1 << 30), (13, 1 << 31), (31, 3 << 30))):
        arrays, expected = arguments(n, i % 2, dtype=dtype, words=words, cancel=cancel, base=base)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        output = tmp_path / f"invoke{i}"
        report = invoke_kernel(compiled, inputs, reference, output, merlin_root=root, python=python)
        assert report["evaluation"]["functional_inputs_preserved"]
        with np.load(output / "evaluation/outputs.npz", allow_pickle=False) as data:
            assert_bits(data["out0"], expected)
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library
