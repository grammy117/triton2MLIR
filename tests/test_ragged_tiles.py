import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.native import execute_native
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared optimizer and native environment required")


def translate(kernel, buffers, constants, grid, root):
    source = root / "kernel.ttir"
    source.write_text(capture_ttir(kernel, {**{b.name: "*fp32" for b in buffers},
                                          **{n: "constexpr" for n in constants}}, constants))
    return convert_shared(source, TensorABI(kernel.__name__, buffers, grid=grid), root / "translation", optimizer=OPT)


@pytest.mark.parametrize("effect,limit,programs", [("write", 53, 4), ("write", 53, 6), ("readwrite", 49, 6)])
def test_dynamic_prefix_tails_preserve_fresh_or_incoming_output(tmp_path, effect, limit, programs):
    from t2m.examples.tensor_kernels import ragged_add
    buffers = (TensorBuffer("x", (53,)), TensorBuffer("out", (53,), effect))
    result = translate(ragged_add, buffers, {"N": 53, "LIMIT": limit, "B": 16}, (programs,), tmp_path)
    x = np.arange(53, dtype=np.float32)
    expected = np.full(53, -13, dtype=np.float32)
    expected[:limit] = 2 * x[:limit]
    inputs = [x] if effect == "write" else [x, np.full(53, -13, dtype=np.float32)]
    (actual,), _ = execute_native(result.mlir_text, inputs, tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual, expected)
    assert '"scf.forall"' in result.mlir_text and '"tensor.pad"' in result.mlir_text
    if effect == "write":
        assert result.trace["memory"]["output_initialization"]["out"]["coverage"] == "disjoint-clipped-affine-prefix-partition"


def test_dynamic_masks_do_not_allow_out_of_bounds_or_uninitialized_fresh_results(tmp_path):
    from t2m.examples.tensor_kernels import ragged_add
    for label, extent, limit, message in [("bounds", 52, 53, "bounds"),
                                          ("holes", 53, 49, "complete overwrite")]:
        root = tmp_path / label
        root.mkdir()
        buffers = (TensorBuffer("x", (extent,)), TensorBuffer("out", (53,), "write"))
        with pytest.raises(ConversionError, match=message):
            translate(ragged_add, buffers, {"N": 53, "LIMIT": limit, "B": 16}, (4,), root)


def test_large_ragged_fresh_result_uses_a_symbolic_partition(tmp_path):
    from t2m.examples.tensor_kernels import ragged_add
    n, block = 2**25 + 13, 256
    buffers = (TensorBuffer("x", (n,)), TensorBuffer("out", (n,), "write"))
    result = translate(ragged_add, buffers, {"N": n, "LIMIT": n, "B": block},
                       ((n + block - 1) // block,), tmp_path)
    assert result.trace["memory"]["output_initialization"]["out"]["coverage"] == "disjoint-clipped-affine-prefix-partition"
    assert len(list(result.module.walk())) < 150


@pytest.mark.parametrize("transposed", [False, True])
def test_ragged_mnk_gemm_keeps_source_tiles_and_k_reuse(tmp_path, transposed):
    from t2m.examples.tensor_kernels import ragged_matmul
    m, n, k = 35, 21, 37
    buffers = (TensorBuffer("x", (m, k)), TensorBuffer("y", (n, k) if transposed else (k, n)),
               TensorBuffer("out", (m, n), "write"))
    result = translate(ragged_matmul, buffers, {"M": m, "N": n, "K": k, "BM": 16, "BN": 16,
                                              "BK": 16, "TRANSPOSED": transposed}, (3, 2), tmp_path)
    rng = np.random.default_rng(28)
    x, y = [rng.normal(size=b.shape).astype("float32") for b in buffers[:2]]
    (actual,), _ = execute_native(result.mlir_text, [x, y], tmp_path / "native", python=NATIVE)
    np.testing.assert_allclose(actual, x @ (y.T if transposed else y), rtol=1e-5, atol=6e-6)
    names = [op.name for op in result.module.walk()]
    assert "scf.forall" in names and "scf.for" in names and "linalg.matmul" in names
    assert result.trace["memory"]["source_loops"][0]["iterations"] == 3
    assert any(e["source_ids"] and e["opcode"] == "tensor.pad" for e in result.trace["emitted_operations"])


def test_dynamic_nonzero_fill_contributes_to_reduction_and_both_branch_results(tmp_path):
    from t2m.examples.tensor_kernels import ragged_fill_reduce
    buffers = (TensorBuffer("x", (53,)), TensorBuffer("out", (53,), "write"))
    result = translate(ragged_fill_reduce, buffers, {"N": 53, "B": 16}, (6,), tmp_path)
    x = np.arange(-45, 8, dtype=np.float32)
    expected = np.empty_like(x)
    for start in range(0, len(x), 16):
        tile = x[start:start + 16]
        total = tile.sum() + 7 * (16 - len(tile))
        expected[start:start + 16] = tile + abs(total)
    (actual,), _ = execute_native(result.mlir_text, [x], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual, expected)
    assert len(result.trace["memory"]["source_conditionals"]) == 1
