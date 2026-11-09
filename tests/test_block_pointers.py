import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.bundle import verify_bundle
from t2m.frontend import capture_ttir
from t2m.native import execute_native
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
from t2m.shared_bundle import write_shared_bundle

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared optimizer and native environment required")


def translate(kernel, buffers, constants, grid, root):
    source = root / "kernel.ttir"
    source.write_text(capture_ttir(kernel, {**{b.name: "*fp32" for b in buffers},
                                          **{n: "constexpr" for n in constants}}, constants))
    return convert_shared(source, TensorABI(kernel.__name__, buffers, grid=grid), root / "translation", optimizer=OPT)


@pytest.mark.parametrize("descriptors", [False, True], ids=["block-pointers", "tensor-descriptors"])
def test_official_block_pointer_pass_preserves_boundary_masks_and_source_intent(tmp_path, descriptors):
    from t2m.examples.tensor_kernels import blockptr_add, descriptor_add
    kernel = descriptor_add if descriptors else blockptr_add
    buffers = (TensorBuffer("x", (53,)), TensorBuffer("out", (53,), "write"))
    result = translate(kernel, buffers, {"N": 53, "B": 16}, (6,), tmp_path)
    x = np.arange(53, dtype=np.float32)
    write_shared_bundle(result, tmp_path / "bundle", [x], [x + 1], work=tmp_path / "native", python=NATIVE)
    assert verify_bundle(tmp_path / "bundle")["materialized_abi"] == {"complete": True, "inputs": 1, "outputs": 1}
    construction = [r for r in result.trace["source_operations"] if r["opcode"] == "tt.make_tensor_ptr"]
    if descriptors:
        assert sum(r["opcode"] == "tt.make_tensor_descriptor" for r in result.trace["source_operations"]) == 2
        assert any(t.get("pass") == "triton-rewrite-tensor-descriptor-to-pointer" for t in result.trace["transformations"])
        return
    assert len(construction) == 2 and all(r["output_types"] == ["!tt.ptr<tensor<16xf32>>"] for r in construction)
    assert result.trace["schema"] == "t2m.shared-tensor-provenance.v2"
    assert "order = array<i32: 0>" in result.source_ttir
    assert "exact-original-source" in result.trace["original_attributes"]
    rewrite = next(t for t in result.trace["transformations"] if t.get("pass") == "triton-rewrite-tensor-pointer")
    assert rewrite["target_backend"] == "not-invoked" and rewrite["source_ids"] == [r["id"] for r in construction]
    assert any(e["normalized_ids"] and e["opcode"] == "tensor.pad" for e in result.trace["emitted_operations"])
    assert all("prov.source_ids" not in op.attributes for op in result.module.walk())


@pytest.mark.parametrize("descriptors", [False, True], ids=["block-pointers", "tensor-descriptors"])
def test_block_pointer_advances_keep_ragged_gemm_loops_and_reuse(tmp_path, descriptors):
    from t2m.examples.tensor_kernels import blockptr_matmul, descriptor_matmul
    kernel = descriptor_matmul if descriptors else blockptr_matmul
    m, n, k = (35, 20, 36) if descriptors else (35, 21, 37)
    buffers = (TensorBuffer("x", (m, k)), TensorBuffer("y", (k, n)), TensorBuffer("out", (m, n), "write"))
    result = translate(kernel, buffers, {"M": m, "N": n, "K": k, "BM": 16, "BN": 16, "BK": 16}, (3, 2), tmp_path)
    rng = np.random.default_rng(29)
    x, y = [rng.normal(size=b.shape).astype("float32") for b in buffers[:2]]
    (actual,), _ = execute_native(result.mlir_text, [x, y], tmp_path / "native", python=NATIVE)
    np.testing.assert_allclose(actual, x @ y, rtol=1e-5, atol=6e-6)
    assert {"scf.forall", "scf.for", "linalg.matmul"} <= {op.name for op in result.module.walk()}
    assert sum(r["opcode"] == "tt.advance" for r in result.trace["source_operations"]) == (0 if descriptors else 2)


@pytest.mark.parametrize("descriptors", [False, True], ids=["block-pointers", "tensor-descriptors"])
def test_block_pointer_boundary_checks_do_not_override_the_tensor_abi(tmp_path, descriptors):
    from t2m.examples.tensor_kernels import blockptr_add, descriptor_add
    buffers = (TensorBuffer("x", (52,)), TensorBuffer("out", (53,), "write"))
    with pytest.raises(ConversionError, match="bounds"):
        translate(descriptor_add if descriptors else blockptr_add, buffers, {"N": 53, "B": 16}, (4,), tmp_path)
