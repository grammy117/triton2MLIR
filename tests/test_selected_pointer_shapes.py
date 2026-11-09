"""Pointer concatenation, split and repeated gathers move selectors with offsets."""

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
    from t2m.examples.selected_pointer_kernels import selected_permute
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    source.write_text(capture_ttir(selected_permute, {"x": "*fp32", "y": "*fp32", "out": "*fp32",
        "n": "i32", "m": "i32", "choice": "i32", "B": "constexpr", "MODE": "constexpr"}, {"B": 8, "MODE": mode}))
    abi = TensorABI("selected_permute", (TensorBuffer("x", ("m",)), TensorBuffer("y", ("m",)),
        TensorBuffer("out", ("n",), "write")), grid=(GridExtent("n", 8 if mode == 1 else 16),),
        runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)), RuntimeScalar("m", "i32", 4, bounds=(0, 63)),
                         RuntimeScalar("choice", "i32", 5, bounds=(0, 3))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT,
                          memory_profile="guarded-buffer", address_policy="source-words")


def arguments(n, choice, mode):
    m = n + 16
    x, y = np.arange(m, dtype=np.float32) - 5, np.arange(m, dtype=np.float32) + 100
    index = np.arange(n)
    if mode == 1:
        take_x = (index + choice) % 2 == 0
        expected = np.where(take_x, x[index], y[m - 1 - index]) - np.float32(2) * np.where(take_x, y[m - 1 - index], x[index])
    else:
        selected = (index % 16 * 6 + choice) % 16 if mode == 2 else index % 16
        source = index // 16 * 16 + selected % 8
        expected = np.where(selected < 8, x[source], y[m - 1 - source])
    return [x, y, *[np.asarray(v, dtype=np.int32) for v in (n, m, choice)]], expected


@pytest.mark.parametrize("mode", [0, 1, 2], ids=["cat", "split", "gather"])
def test_pointer_permutations_keep_buffer_and_offset_correspondence(tmp_path, mode):
    from t2m.native import execute_native
    result = capture(tmp_path, mode)
    assert any(t["kind"] == "selected-pointer-memory" and t["selector"] == "per-lane" for t in result.trace["transformations"])
    for i, n in enumerate((0, 1, 13, 31)):
        arrays, expected = arguments(n, i, mode)
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                     shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("mode", [0, 1, 2], ids=["cat", "split", "gather"])
def test_one_merlin_library_retains_selected_pointer_permutations(tmp_path, mode):
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
