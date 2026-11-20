"""Typed while state keeps its full word domain before address promotion."""

import json
import os

import numpy as np
import pytest
from xdsl.dialects import builtin

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not os.environ.get("TRITON_SHARED_OPT"), reason="existing shared optimizer required")
CONFIGURATIONS = (("prefix", 32), ("wrapped", 32), ("wrapped", 64), ("direct", 64))


def capture(root, kind="prefix", bits=32, address="wrap"):
    from t2m.examples.while_index_kernels import prefix_while, wrapped_while

    source = root / "source.ttir"
    pointers = {name: "*i64" for name in ("x", "state", "before", "after")}
    word = "i" + str(bits)
    if kind == "prefix":
        source.write_text(capture_ttir(prefix_while, {**pointers, "n": "i32"}, {}))
        scalar = (RuntimeScalar("n", "i32", 4, bounds=(0, 31)),)
        extent = "n"
    else:
        source.write_text(capture_ttir(wrapped_while, {**pointers, "start": word, "steps": "i32",
            "delta": word, "ADDRESS": "constexpr"}, {"ADDRESS": "direct" if kind == "direct" else address}))
        scalar = (RuntimeScalar("start", word, 4), RuntimeScalar("steps", "i32", 5, bounds=(0, 5)),
                  RuntimeScalar("delta", word, 6))
        extent = 16
    abi = TensorABI("prefix_while" if kind == "prefix" else "wrapped_while",
        (TensorBuffer("x", (extent,), "read", "i64"), TensorBuffer("state", (1,), "readwrite", "i64"),
         TensorBuffer("before", (1,), "write", "i64"), TensorBuffer("after", (1,), "write", "i64")),
        runtime_scalars=scalar)
    return source, abi


@pytest.mark.parametrize("kind,bits", CONFIGURATIONS)
def test_actual_while_indices_reach_tensor_memory_with_original_word_state(tmp_path, kind, bits):
    source, abi = capture(tmp_path, kind, bits)
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    assert any(op.name == "scf.while" for op in result.module.walk())
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    loop = next(op for op in result.module.walk() if op.name == "scf.while")
    words = sorted(arg.type.bitwidth for arg in loop.before_region.block.args if isinstance(arg.type, builtin.IntegerType))
    assert words == ([32] if kind == "prefix" else sorted([bits, 32]))
    assert any(row["kind"] == "source-while-word-ranges" for row in result.trace["transformations"])


@pytest.mark.parametrize("address,expected", (("add", "overflow"), ("divide", "zero divisor"), ("narrow", "overflow")))
def test_full_while_word_domains_do_not_prove_unsafe_address_arithmetic(tmp_path, address, expected):
    source, abi = capture(tmp_path, "wrapped", 64, address)
    with pytest.raises(ConversionError, match=expected):
        convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def arguments(kind, bits, index):
    def word(value, width):
        return (value + (1 << (width - 1))) % (1 << width) - (1 << (width - 1))

    old = np.asarray([(1 << 56) + 1], np.int64)
    n = (0, 1, 8, 31, 17)[index] if kind == "prefix" else 16
    x = np.asarray([(i + 1) ** 3 * 13 + (1 << 55) for i in range(n)], np.int64)
    state = int(old[0])
    if kind == "prefix":
        for i in range(n):
            state ^= word(int(x[i]) + i, 64)
        after = state
        controls = [np.asarray(n, np.int32)]
    else:
        edge = 1 << (bits - 1)
        start, steps, delta = ((edge - 2, 3, 3), (-edge + 1, 5, -3), (edge - 1, 0, edge - 1),
                               (-7, 1, 0), (0, 5, -edge))[index]
        if kind == "direct":
            start, steps, delta = ((3, 3, 2), (0, 5, 3), (-1, 0, edge - 1), (7, 1, 0), (14, 2, -7))[index]
        i = start
        for _ in range(steps):
            state ^= int(x[i if kind == "direct" else i % 15])
            i = word(i + delta, bits)
        after = state ^ int(x[i % 15])
        dtype = np.int32 if bits == 32 else np.int64
        controls = [np.asarray(start, dtype), np.asarray(steps, np.int32), np.asarray(delta, dtype)]
    return [x, old, *controls], [np.asarray([state], np.int64), old.copy(), np.asarray([after], np.int64)]


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native toolchain required")
@pytest.mark.parametrize("kind,bits", CONFIGURATIONS)
def test_native_while_addresses_keep_signed_words_wraparound_and_final_state(tmp_path, kind, bits):
    from t2m.native import execute_native

    source, abi = capture(tmp_path, kind, bits)
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    for index in range(5):
        inputs, expected = arguments(kind, bits, index)
        actual, _ = execute_native(result.mlir_text, inputs, tmp_path / f"native-{index}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=abi.runtime_shapes())
        for observed, wanted in zip(actual, expected, strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())


@pytest.fixture(scope="module", params=CONFIGURATIONS)
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(name) for name in ("T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing native/Merlin tools required")
    kind, bits = request.param
    root = tmp_path_factory.mktemp(f"while-indices-{kind}-{bits}")
    source, abi = capture(root, kind, bits)
    result = convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    write_shared_bundle(result, root / "capture", *arguments(kind, bits, 0), work=root / "native",
        python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"])
    return root, kind, bits


def invocation(root, kind, bits, index, name):
    inputs, expected = arguments(kind, bits, index)
    values, reference = root / f"inputs-{name}.npz", root / f"reference-{name}.npz"
    np.savez(values, **{f"in{i}": value for i, value in enumerate(inputs)})
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
    return values, reference


def test_one_merlin_library_retains_independent_while_indices_and_every_state(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import verify_kernel_qualification

    root, kind, bits = compiled
    library, receipt = _record(root / "merlin/host/model.so"), verify_kernel_qualification(root / "merlin")
    for index in range(5):
        inputs, reference = invocation(root, kind, bits, index, str(index))
        out = root / f"invoke-{index}"
        report = invoke_kernel(root / "merlin", inputs, reference, out,
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(out)
    assert _record(root / "merlin/host/model.so") == library and verify_kernel_qualification(root / "merlin") == receipt


def test_actual_serial_images_keep_word_overflow_and_zero_trips(compiled):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    root, kind, bits = compiled
    for index in (0, 2):
        inputs, reference = invocation(root, kind, bits, index, "target-" + str(index))
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=inputs, reference=reference, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 3
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"],
                        python=os.environ["T2M_MERLIN_PYTHON"]) == report


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native tools required")
@pytest.mark.parametrize("bits", (32, 64))
def test_full_while_words_keep_actual_active_bounds_guards(tmp_path, bits):
    from t2m.native import execute_native

    source, abi = capture(tmp_path, "direct", bits)
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    # A negative body index is harmless when that body never executes. Its
    # final normalized index still supplies the original post-loop observation.
    values, expected = arguments("direct", bits, 2)
    actual, _ = execute_native(result.mlir_text, values, tmp_path / "native-zero", python=os.environ["T2M_NATIVE_PYTHON"])
    for observed, wanted in zip(actual, expected, strict=True):
        assert observed.tobytes() == wanted.tobytes()
    failures = []
    for index, offset in enumerate((-1, 16, *(((1 << 32) + 3,) if bits == 64 else ()))):
        bad = [*values]
        bad[2] = np.asarray(offset, np.int32 if bits == 32 else np.int64)
        bad[3] = np.asarray(1, np.int32)
        out = tmp_path / f"aborted-{index}"
        with pytest.raises(ValueError, match="native host execution failed") as failure:
            execute_native(result.mlir_text, bad, out, python=os.environ["T2M_NATIVE_PYTHON"])
        assert not (out / "outputs.npz").exists()
        assert "exit -6" in str(failure.value)
        failures.append({"offset": offset, "error": str(failure.value)})
    (tmp_path / "bounds-aborts.json").write_text(json.dumps(failures, indent=2) + "\n")
