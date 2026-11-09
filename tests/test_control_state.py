"""Computed reduction state keeps original init reads through real Merlin."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
CASES = ((0, 5, 3), (1, 0, 2), (7, 1, 1), (9, 3, 1), (13, 5, 3))


def capture(root):
    from t2m.examples.control_state_kernels import conditional_reduction
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    source.write_text(capture_ttir(conditional_reduction, {"x": "*fp32", "out": "*fp32", "n": "i32", "steps": "i32",
        "choice": "i32", "B": "constexpr", "R": "constexpr"}, {"B": 8, "R": 4}))
    abi = TensorABI("conditional_reduction", (TensorBuffer("x", (4, "n")), TensorBuffer("out", ("n",), "write")),
        grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 13)),
            RuntimeScalar("steps", "i32", 3, bounds=(0, 5)), RuntimeScalar("choice", "i32", 4, bounds=(0, 3))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT,
                          memory_profile="guarded-buffer", address_policy="source-words")


def arguments(n, steps, choice):
    x = (np.arange(4 * n, dtype=np.float32) - 10).reshape(4, n)
    acc = np.full(n, 2.5, dtype=np.float32)
    for step in range(steps):
        acc = x.sum(axis=0) if (step + choice) % 2 == 0 else (x + acc).sum(axis=0)
    return [x, *[np.asarray(v, dtype=np.int32) for v in (n, steps, choice)]], acc


def test_conditional_reduction_retains_original_state_and_numeric_order(tmp_path):
    from t2m.native import execute_native
    result = capture(tmp_path)
    assert "scf.while" in result.mlir_text and "scf.if" in result.mlir_text and "scf.forall" in result.mlir_text
    assert any(op["opcode"] == "tt.reduce" for op in result.trace["source_operations"])
    for i, case in enumerate(CASES):
        arrays, expected = arguments(*case)
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                     shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
def test_merlin_conditional_reduction_library_preserves_inputs_and_initial_state(tmp_path):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result = capture(tmp_path)
    arrays, expected = arguments(9, 3, 2)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    verify_bundle(bundle)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, case in enumerate(CASES):
        arrays, expected = arguments(*case)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        output = tmp_path / f"invoke{i}"
        report = invoke_kernel(compiled, inputs, reference, output, merlin_root=root, python=python)
        verify_invocation(output)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library
