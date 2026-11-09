"""Runtime admission for indirect stores retains the upstream memory payload."""

import json
import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def convert(root, *, fresh=False, repeat=1, read_output=False, store_input=False, index_dtype="i32", scalar=False):
    from t2m.examples.tensor_kernels import indirect_store, scalar_indirect_store
    from t2m.frontend import capture_ttir
    path = root / "source.ttir"
    kernel = scalar_indirect_store if scalar else indirect_store
    constants = {"REPEAT": repeat, "READ_OUTPUT": read_output, "STORE_INPUT": store_input}
    if not scalar:
        constants["B"] = 8
    path.write_text(capture_ttir(kernel,
        {"x": "*fp32", "indices": "*" + index_dtype, "out": "*fp32", "n": "i32", "m": "i32",
         **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", ("n",)), TensorBuffer("indices", ("n",), dtype=index_dtype),
        TensorBuffer("out", ("m",), "write" if fresh else "readwrite")), grid=(GridExtent("n", 1 if scalar else 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)), RuntimeScalar("m", "i32", 4, bounds=(0, 31))))
    return convert_shared(path, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer"), abi


def arguments(indices, m, *, fresh=False, repeat=1, dtype=np.int32):
    indices = np.asarray(indices, dtype=dtype)
    n = len(indices)
    x, old = np.arange(n, dtype=np.float32), np.arange(m, dtype=np.float32) + 100
    expected = np.empty(m, dtype=np.float32) if fresh else old.copy()
    active = indices >= 0
    expected[indices[active]] = x[active] + np.float32(repeat)
    arrays = [x, indices, *([] if fresh else [old]), np.asarray(n, dtype=np.int32), np.asarray(m, dtype=np.int32)]
    return arrays, expected


@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("index_dtype", ["i32", "i64"])
@pytest.mark.parametrize("scalar", [False, True])
def test_indirect_permutations_repeated_writes_and_masked_indices(tmp_path, fresh, index_dtype, scalar):
    from t2m.native import execute_native
    result, abi = convert(tmp_path, fresh=fresh, repeat=3, index_dtype=index_dtype, scalar=scalar)
    indices = [7, 0, 6, 2, 5, 1, 4, 3, -(1 << 30)]
    arrays, expected = arguments(indices, 8, fresh=fresh, repeat=3, dtype={"i32": np.int32, "i64": np.int64}[index_dtype])
    (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=abi.runtime_shapes())
    np.testing.assert_array_equal(actual, expected)
    assert result.trace["profile"] == "shared-guarded-buffer"
    assert "memory-witness.mlir" not in result.auxiliary_artifacts
    assert "memref.atomic_rmw" in result.mlir_text and "memref.store" in result.mlir_text
    assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    assert any(op.name == "scf.for" for op in result.module.walk())
    admission = json.loads(result.auxiliary_artifacts["memory-admission.json"])
    assert bool(admission["scalar_normalizations"]) == scalar


@pytest.mark.parametrize("indices,m,fresh", [([2, 2], 4, False), ([0, 1, 2, 3, 4, 5, 6, 7, 0], 9, False),
    ([4], 4, False), ([0], 0, False), ([0, 2], 4, True)])
@pytest.mark.parametrize("scalar", [False, True])
def test_conflicting_oob_and_unwritten_outputs_fail_executable_guards(tmp_path, indices, m, fresh, scalar):
    from t2m.native import execute_native
    result, abi = convert(tmp_path, fresh=fresh, scalar=scalar)
    arrays = [np.arange(len(indices), dtype=np.float32), np.asarray(indices, dtype=np.int32),
              *([] if fresh else [np.zeros(m, dtype=np.float32)]), np.asarray(len(indices), dtype=np.int32), np.asarray(m, dtype=np.int32)]
    with pytest.raises(ValueError, match=r"native host execution failed \(exit -6\)"):
        execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=abi.runtime_shapes())


@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("scalar", [False, True])
def test_empty_output_and_inactive_extreme_indices_are_valid(tmp_path, fresh, scalar):
    from t2m.native import execute_native
    result, abi = convert(tmp_path, fresh=fresh, scalar=scalar)
    arrays, expected = arguments([-2147483648, -1], 0, fresh=fresh)
    (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=abi.runtime_shapes())
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("options,reason", [({"read_output": True}, "writable-buffer reads"),
    ({"store_input": True}, "read-only buffer")])
def test_effects_outside_guarded_private_output_contract_refuse(tmp_path, options, reason):
    with pytest.raises(ConversionError, match=reason) as caught:
        convert(tmp_path, **options)
    assert caught.value.source_id is not None
    assert not (tmp_path / "translation/memory-ir").exists()


def test_rank_two_structured_access_guards_match_upstream_pointer_offsets(tmp_path):
    from t2m.cli import shared_demo
    from t2m.bundle import verify_bundle
    capture = tmp_path / "capture"
    shared_demo(capture, kernel="matrix-add", optimizer=OPT, python=NATIVE, memory_profile="guarded-buffer")
    assert verify_bundle(capture)["profile"] == "shared-guarded-buffer"


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("scalar", [False, True])
def test_indirect_store_package_reuses_one_actual_merlin_library(tmp_path, scalar):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.memory_contract import GUARDED_FILES
    from t2m.shared_bundle import write_shared_bundle
    result, _ = convert(tmp_path, repeat=3, scalar=scalar)
    arrays, expected = arguments([7, 0, 6, -1, 5, 1, 4, 3, -99], 11, repeat=3)
    capture, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, capture, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    receipt = verify_bundle(capture)
    assert GUARDED_FILES <= receipt["artifacts"].keys()
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(capture, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    qualified, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    signature = json.loads((compiled / "compiler/signature.json").read_text())
    assert signature["guard_storage_bytes_per_result_element"] == 16
    for ordinal, (indices, m) in enumerate((([], 0), ([-1], 4), ([3, 0], 4), ([11, 2, 9, 0, -1, 8, 6, 1, 7], 13))):
        arrays, expected = arguments(indices, m, repeat=3)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        out = tmp_path / ("invoke" + str(ordinal))
        report = invoke_kernel(compiled, inputs, reference, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == qualified and _record(compiled / "host/model.so") == library
    raw = (capture / "memory-admission.json").read_bytes()
    (capture / "memory-admission.json").write_bytes(raw + b"\n")
    receipt["artifacts"]["memory-admission.json"] = _record(capture / "memory-admission.json")
    (capture / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="memory evidence differs"):
        verify_bundle(capture)
