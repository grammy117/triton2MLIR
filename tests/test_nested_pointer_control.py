"""Branch-selected pointer state retains iteration structure through Merlin."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared, opcode, parse_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
CASES = ((0, 5, 3), (1, 0, 2), (7, 1, 0), (9, 3, 1), (13, 5, 3))


def capture(root, *, deep=False, for_loop=False, rows=3):
    from t2m.examples.pointer_control_kernels import nested_pointer_sum
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    source.write_text(capture_ttir(nested_pointer_sum, {"x": "*fp32", "out": "*fp32", "n": "i32", "steps": "i32",
        "choice": "i32", "B": "constexpr", "DEEP": "constexpr", "FOR_LOOP": "constexpr"},
        {"B": 8, "DEEP": deep, "FOR_LOOP": for_loop}))
    abi = TensorABI("nested_pointer_sum", (TensorBuffer("x", (rows, "n")), TensorBuffer("out", ("n",), "write")),
        grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 13)),
            RuntimeScalar("steps", "i32", 3, bounds=(0, 5)), RuntimeScalar("choice", "i32", 4, bounds=(0, 3))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT,
                          memory_profile="guarded-buffer", address_policy="source-words")


def arguments(n, steps, choice, *, deep=False, rows=3):
    x = (np.arange(rows * n, dtype=np.float32) - 10).reshape(rows, n)
    arrays = [x, *[np.asarray(v, dtype=np.int32) for v in (n, steps, choice)]]
    expected = np.zeros(n, dtype=np.float32)
    if n:
        selected = 0
        for step in range(steps):
            selected = 0 if (step + choice) % 2 == 0 else 2 if deep and choice > 1 else 1
            expected += x[selected]
        expected += x[selected]
    return arrays, expected


@pytest.mark.parametrize("deep", [False, True])
@pytest.mark.parametrize("for_loop", [False, True])
def test_nested_pointer_regions_and_final_values_execute_natively(tmp_path, deep, for_loop):
    from t2m.native import execute_native
    result = capture(tmp_path, deep=deep, for_loop=for_loop)
    module = parse_shared(result.mlir_text)
    assert any(opcode(op) == ("scf.for" if for_loop else "scf.while") for op in module.walk())
    assert "scf.if" in result.mlir_text and "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    assert any(r["opcode"] == "tt.call" for r in result.trace["source_operations"])
    for i, case in enumerate(CASES):
        arrays, expected = arguments(*case, deep=deep)
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}",
                                     python=NATIVE, shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual, expected)


def test_only_taken_branch_addresses_are_checked(tmp_path):
    from t2m.native import execute_native
    result = capture(tmp_path, deep=True, rows=2)
    arrays, expected = arguments(9, 1, 2, deep=True, rows=2)
    (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / "native-valid", python=NATIVE,
                                 shape_abi=result.spec.runtime_shapes())
    np.testing.assert_array_equal(actual, expected)
    arrays[2] = np.asarray(2, dtype=np.int32)
    with pytest.raises(ValueError, match="exit -6"):
        execute_native(result.mlir_text, arrays, tmp_path / "native-invalid", python=NATIVE,
                       shape_abi=result.spec.runtime_shapes())


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("deep", [False, True])
@pytest.mark.parametrize("for_loop", [False, True])
def test_one_merlin_library_preserves_nested_pointer_state(tmp_path, deep, for_loop):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result = capture(tmp_path, deep=deep, for_loop=for_loop)
    arrays, expected = arguments(9, 3, 2, deep=deep)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    verify_bundle(bundle)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, case in enumerate(CASES):
        arrays, expected = arguments(*case, deep=deep)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        output = tmp_path / f"invoke{i}"
        report = invoke_kernel(compiled, inputs, reference, output, merlin_root=root, python=python)
        verify_invocation(output)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library
