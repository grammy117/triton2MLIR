"""A single logical pointer element becomes scalar without observing an address."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def capture(root, mode):
    from t2m.examples.selected_pointer_kernels import selected_unsplat
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    source.write_text(capture_ttir(selected_unsplat, {"x": "*fp32", "y": "*fp32", "out": "*fp32",
        "n": "i32", "choice": "i32", "MODE": "constexpr"}, {"MODE": mode}))
    abi = TensorABI("selected_unsplat", (TensorBuffer("x", ("n",)), TensorBuffer("y", ("n",)),
        TensorBuffer("out", ("n",), "write")), grid=(GridExtent("n", 1),), runtime_scalars=(
        RuntimeScalar("n", "i32", 3, bounds=(0, 31)), RuntimeScalar("choice", "i32", 4, bounds=(0, 3))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT,
                          memory_profile="guarded-buffer", address_policy="source-words")


def arguments(n, choice, mode):
    index = np.arange(n)
    x, y = index.astype(np.float32) - 5, index.astype(np.float32) + 100
    take_x = np.ones(n, dtype=bool) if mode == 0 else (np.full(n, choice == 0) if mode == 1 else (index + choice) % 2 == 0)
    return [x, y, np.asarray(n, dtype=np.int32), np.asarray(choice, dtype=np.int32)], np.where(take_x, x, y[::-1])


@pytest.mark.parametrize("mode", [0, 1, 2], ids=["same-base", "branch", "lane-select"])
def test_pointer_unsplat_keeps_logical_base_and_offset(tmp_path, mode):
    from t2m.native import execute_native
    result = capture(tmp_path, mode)
    assert any(r["opcode"] == "tt.unsplat" for r in result.trace["source_operations"])
    for i, n in enumerate((0, 1, 13, 31)):
        arrays, expected = arguments(n, i, mode)
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                     shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("mode", [0, 1, 2], ids=["same-base", "branch", "lane-select"])
def test_one_merlin_library_retains_scalarized_selected_pointer(tmp_path, mode):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result = capture(tmp_path, mode)
    arrays, expected = arguments(13, 1, mode)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, n in enumerate((0, 1, 13, 31)):
        arrays, expected = arguments(n, i, mode)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        report = invoke_kernel(compiled, inputs, reference, tmp_path / f"invoke{i}", merlin_root=root, python=python)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library
