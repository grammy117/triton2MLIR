"""A flat transfer retains multidimensional computation and actual predicates."""

import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON",
    "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/native/Merlin tools required")


def capture(root, dtype, profile, partial=False):
    from t2m.examples.linear_tile_kernels import linear_tile_update

    root.mkdir()
    source = root / "source.ttir"
    source.write_text(capture_ttir(linear_tile_update, {"x": "*fp32" if dtype == "f32" else "*" + dtype,
        "out": "*fp32" if dtype == "f32" else "*" + dtype, "n": "i32", "B": "constexpr", "PARTIAL": "constexpr"},
        {"B": 8, "PARTIAL": partial}))
    abi = TensorABI("linear_tile_update", (TensorBuffer("x", ("n",), dtype=dtype),
        TensorBuffer("out", ("n",), "write", dtype)), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 67)),))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
                          memory_profile=profile)


@pytest.mark.parametrize("dtype", ("i32", "i64", "f32"))
@pytest.mark.parametrize("profile", ("tensor", "buffer"))
def test_rank_two_tiles_keep_all_flat_allocation_values_and_runtime_tails(tmp_path, dtype, profile):
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    result = capture(tmp_path / "source", dtype, profile)
    tensor_program = (result.mlir_text if profile == "tensor"
                      else result.auxiliary_artifacts["memory-witness.mlir"])
    assert "tensor.collapse_shape" in tensor_program
    assert "tensor<2x4x" in tensor_program
    np_type = {"i32": np.int32, "i64": np.int64, "f32": np.float32}[dtype]

    def values(n):
        data = np.arange(n, dtype=np_type) - 11
        if dtype == "i64":
            data += 1 << 60
        return [data, np.asarray(n, np.int32)], [data * 2 + 1]

    arrays, expected = values(13)
    write_shared_bundle(result, tmp_path / "capture", arrays, expected, work=tmp_path / "native",
                        python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(tmp_path / "capture", tmp_path / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"])
    for n in (0, 1, 31):
        arrays, expected = values(n)
        inputs, reference = tmp_path / f"inputs-{n}.npz", tmp_path / f"reference-{n}.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, out0=expected[0])
        invoke_kernel(tmp_path / "merlin", inputs, reference, tmp_path / f"invoke-{n}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(tmp_path / f"invoke-{n}")


@pytest.mark.parametrize("profile", ("tensor", "buffer"))
def test_data_predicate_cannot_be_erased_by_flattening_a_fresh_write(tmp_path, profile):
    with pytest.raises(ConversionError, match="initial|complete|prior|read"):
        capture(tmp_path / "source", "i32", profile, partial=True)
