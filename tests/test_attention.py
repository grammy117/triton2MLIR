"""Online state and tile contractions survive the actual Merlin interface."""

import os

import numpy as np
import pytest

from xdsl.dialects.builtin import TensorType

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
CASES = ((0, 0), (0, 19), (1, 0), (1, 1), (16, 16), (17, 19), (35, 37), (33, 3))
RTOL, ATOL = 2e-5, 2e-6


def capture(root):
    from t2m.examples.attention_kernels import online_attention
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    source.write_text(capture_ttir(online_attention, {**{name: "*fp32" for name in ("q", "k", "v", "out")},
        "m": "i32", "n": "i32", "D": "constexpr", "BM": "constexpr", "BN": "constexpr"},
        {"D": 16, "BM": 16, "BN": 16}))
    abi = TensorABI("online_attention", (TensorBuffer("q", ("m", 16)), TensorBuffer("k", ("n", 16)),
        TensorBuffer("v", ("n", 16)), TensorBuffer("out", ("m", 16), "write")), grid=(GridExtent("m", 16),),
        runtime_scalars=(RuntimeScalar("m", "i32", 4, bounds=(0, 35)), RuntimeScalar("n", "i32", 5, bounds=(0, 37))),
        intent="Tiled online softmax; running maximum, normalizer and accumulator; zero output for zero keys.")
    return convert_shared(source, abi, root / "translation", optimizer=OPT,
                          memory_profile="guarded-buffer", address_policy="source-words")


def arguments(m, n):
    rng = np.random.default_rng(73 + 100 * m + n)
    q = rng.normal(size=(m, 16)).astype(np.float32)
    k = rng.normal(size=(n, 16)).astype(np.float32)
    v = rng.normal(size=(n, 16)).astype(np.float32)
    expected = np.zeros((m, 16), dtype=np.float32)
    if m and n:
        # Independent dense float64 oracle; this is not emitted into the kernel.
        scores = (q.astype(np.float64) @ k.astype(np.float64).T) * 0.25
        probability = np.exp(scores - scores.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        expected = (probability @ v.astype(np.float64)).astype(np.float32)
    return [q, k, v, np.asarray(m, dtype=np.int32), np.asarray(n, dtype=np.int32)], expected


def test_tiled_attention_keeps_online_state_and_executes_independent_shapes(tmp_path):
    from t2m.native import execute_native
    result = capture(tmp_path)
    source = result.trace["source_operations"]
    assert sum(op["opcode"] == "tt.dot" for op in source) == 2
    assert any(t["kind"] == "restore-source-contraction-accumulator" for t in result.trace["transformations"])
    assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    loops = [op for op in result.module.walk() if op.name == "scf.for" and len(op.results) == 3]
    assert any(sorted(v.type.get_shape() for v in op.results if isinstance(v.type, TensorType))
               == [(16,), (16,), (16, 16)] for op in loops)
    assert sum(op.name == "linalg.matmul" for op in result.module.walk()) == 2
    for op in result.module.walk():
        for value in (*op.operands, *op.results):
            if isinstance(value.type, TensorType) and len(value.type.get_shape()) == 2:
                # No whole query-by-key score tensor: one dimension is a tile/channel.
                assert 16 in value.type.get_shape()
    for i, shape in enumerate(CASES):
        arrays, expected = arguments(*shape)
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                     shape_abi=result.spec.runtime_shapes())
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
def test_one_merlin_attention_library_retains_tiling_and_runtime_softmax_state(tmp_path):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result = capture(tmp_path)
    arrays, expected = arguments(17, 19)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, [expected], work=tmp_path / "native", python=NATIVE, rtol=RTOL, atol=ATOL)
    verify_bundle(bundle)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang, rtol=RTOL, atol=ATOL)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, shape in enumerate(CASES):
        arrays, expected = arguments(*shape)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        output = tmp_path / f"invoke{i}"
        report = invoke_kernel(compiled, inputs, reference, output, merlin_root=root, python=python, rtol=RTOL, atol=ATOL)
        verify_invocation(output)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library


def test_online_attention_demo_packages_the_tiled_source_and_evidence(tmp_path):
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    output = tmp_path / "demo"
    assert main(["shared-demo", "--kernel", "online-attention", "--out", str(output), "--optimizer", OPT,
                 "--python", NATIVE, "--memory-profile", "guarded-buffer", "--address-policy", "source-words"]) == 0
    receipt = verify_bundle(output)
    assert "word-input.mlir" in receipt["artifacts"]
    assert "linalg.matmul" in (output / "model.mlir").read_text()
