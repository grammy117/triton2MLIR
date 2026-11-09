"""Exact high words and Philox agree with independent wide-integer references."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
PROFILES = ["tensor", "buffer", "guarded-buffer"]
NATIVE_REQUIRED = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


@pytest.mark.parametrize("flags", ["nsw", "nuw", "nsw, nuw"])
def test_poison_producing_truncation_flags_remain_a_located_refusal(flags):
    from t2m.arith_compat import normalize_truncation_defaults
    from t2m.diagnostics import ConversionError
    from t2m.shared_tensor import parse_shared, _source_ids
    module = parse_shared(f'''builtin.module {{ func.func @cast(%x: i64) -> i32 {{
      %y = "arith.trunci"(%x) <{{overflowFlags = #arith.overflow<{flags}>}}> : (i64) -> i32 loc("ttir.op.cast")
      func.return %y : i32
    }} }}''')
    with pytest.raises(ConversionError, match="poison-producing") as error:
        normalize_truncation_defaults(module, _source_ids)
    assert error.value.source_id == "ttir.op.cast" and error.value.opcode == "arith.trunci"


def test_explicit_neutral_truncation_default_preserves_modular_semantics():
    from t2m.arith_compat import normalize_truncation_defaults
    from t2m.shared_tensor import parse_shared, _source_ids
    module = parse_shared('''builtin.module { func.func @cast(%x: i64) -> i32 {
      %y = "arith.trunci"(%x) <{overflowFlags = #arith.overflow<none>}> : (i64) -> i32 loc("ttir.op.cast")
      func.return %y : i32
    } }''')
    reports = normalize_truncation_defaults(module, _source_ids)
    assert reports[0]["source_default"] == "overflow<none>"
    assert reports[0]["source_ids"] == ["ttir.op.cast"]
    module.verify()
    assert not normalize_truncation_defaults(module, _source_ids)


def capture(root, profile, width=32, scalar=False, random=False):
    from t2m.examples.integer_product_kernels import high_product_values, scalar_high_product, philox_words
    from t2m.frontend import capture_ttir
    dtype = f"i{width}"
    if random:
        kernel = philox_words
        buffers = tuple(TensorBuffer(f"out{i}", ("n",), "write", dtype="i32") for i in range(4))
        scalars = (RuntimeScalar("n", "i32", 4, bounds=(0, 31)), RuntimeScalar("seed", "i64", 5), RuntimeScalar("counter", "i64", 6))
    elif scalar:
        kernel = scalar_high_product
        buffers = (TensorBuffer("out", ("n",), "write", dtype=dtype),)
        scalars = (RuntimeScalar("a", dtype, 1), RuntimeScalar("b", dtype, 2), RuntimeScalar("n", "i32", 3, bounds=(0, 31)))
    else:
        kernel = high_product_values
        buffers = (TensorBuffer("x", ("n",), dtype=dtype), TensorBuffer("y", ("n",), dtype=dtype),
                   TensorBuffer("out", ("n",), "write", dtype=dtype))
        scalars = (RuntimeScalar("n", "i32", 3, bounds=(0, 31)),)
    path = root / "source.ttir"
    signature = {**{b.name: f"*{b.dtype}" for b in buffers}, **{s.name: s.dtype for s in scalars}, "B": "constexpr"}
    path.write_text(capture_ttir(kernel, signature, {"B": 8}))
    abi = TensorABI(kernel.__name__, buffers, grid=(GridExtent("n", 8),), runtime_scalars=scalars)
    return convert_shared(path, abi, root / "translation", optimizer=OPT, memory_profile=profile,
                          address_policy="source-words" if profile == "guarded-buffer" else "structured")


def product_cases(width, scalar):
    unsigned, signed = getattr(np, f"uint{width}"), getattr(np, f"int{width}")
    mask = (1 << width) - 1
    words = np.array([0, 1, mask, mask - 1, 1 << (width - 1), (1 << (width - 1)) - 1, 0x12345678], dtype=unsigned)
    cases = []
    for n in (0, 1, 13, 31):
        if scalar:
            a = np.asarray(words[n % len(words)]).view(signed)
            b = np.asarray(words[(n + 3) % len(words)]).view(signed)
            high = ((int(a) & mask) * (int(b) & mask)) >> width
            expected = np.full(n, high, dtype=unsigned).view(signed)
            arrays = [a, b, np.asarray(n, np.int32)]
        else:
            a = np.resize(words, n).view(signed).copy()
            b = np.resize(np.roll(words, 3), n).view(signed).copy()
            expected = np.array([((int(x) & mask) * (int(y) & mask)) >> width for x, y in zip(a, b)], dtype=unsigned).view(signed)
            arrays = [a, b, np.asarray(n, np.int32)]
        cases.append((arrays, [expected]))
    return cases


def philox_reference(counter, key):
    """Python unbounded products, checked against Random123's known vectors."""
    mask = (1 << 32) - 1
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for _ in range(10):
        p0, p2 = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (p2 >> 32) ^ c1 ^ k0, p2 & mask, (p0 >> 32) ^ c3 ^ k1, p0 & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return c0, c1, c2, c3


def test_independent_philox_oracle_matches_published_known_vectors():
    # DEShawResearch/random123, tests/kat_vectors, philox4x32 10.
    # https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    assert philox_reference((0, 0, 0, 0), (0, 0)) == (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    assert philox_reference((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2) == (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)
    assert philox_reference((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0)) == (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)


def random_cases():
    result = []
    for n, seed, counter in ((0, 0, 0), (1, 0, 0), (13, 2**32 + 19, 2**32 - 3), (31, -1, -9)):
        rows = []
        for lane in range(n):
            offset = (counter + lane) % (1 << 64)
            rows.append(philox_reference((offset & 0xFFFFFFFF, offset >> 32, 0, 0), (seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF)))
        outputs = np.asarray(rows, dtype=np.uint32).reshape(n, 4).view(np.int32)
        result.append(([np.asarray(n, np.int32), np.asarray(seed, np.int64), np.asarray(counter, np.int64)],
                       [outputs[:, i].copy() for i in range(4)]))
    return result


def run_native(root, result, cases):
    from t2m.native import execute_native
    for i, (arrays, expected) in enumerate(cases):
        actual, _ = execute_native(result.mlir_text, arrays, root / f"native{i}", python=NATIVE, shape_abi=result.spec.runtime_shapes())
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(value, reference)


def replay_library(root, result, cases):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    bundle, compiled = root / "capture", root / "merlin"
    arrays, expected = cases[-1]
    write_shared_bundle(result, bundle, arrays, expected, work=root / "native", python=NATIVE)
    merlin, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=merlin, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, (arrays, expected) in enumerate(cases):
        inputs, reference = root / "inputs.npz", root / "reference.npz"
        np.savez(inputs, **{f"in{j}": a for j, a in enumerate(arrays)})
        np.savez(reference, **{f"out{j}": a for j, a in enumerate(expected)})
        report = invoke_kernel(compiled, inputs, reference, root / f"invoke{i}", merlin_root=merlin, python=python)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library


@NATIVE_REQUIRED
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_unsigned_high_products_retain_source_words(tmp_path, profile, width, scalar):
    result = capture(tmp_path, profile, width, scalar)
    assert any(t["kind"] == "ttir-unsigned-high-product" for t in result.trace["transformations"])
    assert any(r["opcode"] == "tt.mulhiui" for r in result.trace["source_operations"])
    assert "mului_extended" in result.mlir_text
    run_native(tmp_path, result, product_cases(width, scalar))


@NATIVE_REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_merlin_reuses_high_product_library_across_words_and_shapes(tmp_path, profile, width, scalar):
    replay_library(tmp_path, capture(tmp_path, profile, width, scalar), product_cases(width, scalar))


@NATIVE_REQUIRED
@pytest.mark.parametrize("profile", PROFILES)
def test_original_triton_philox_keeps_integer_rounds(tmp_path, profile):
    result = capture(tmp_path, profile, random=True)
    reports = [t for t in result.trace["transformations"] if t["kind"] == "ttir-unsigned-high-product"]
    assert len(reports) == 20
    run_native(tmp_path, result, random_cases())


@NATIVE_REQUIRED
@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("profile", PROFILES)
def test_merlin_philox_retains_seed_counter_and_all_result_words(tmp_path, profile):
    replay_library(tmp_path, capture(tmp_path, profile, random=True), random_cases())
