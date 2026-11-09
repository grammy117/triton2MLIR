"""Mutable observations keep source order, initialization and grid independence."""

import json
import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")


def recurrence(root, *, scalar=False, fresh=False, initialize=False, dtype="i32"):
    from t2m.examples.tensor_kernels import indirect_recurrence
    from t2m.frontend import capture_ttir
    path = root / "source.ttir"
    constants = {"B": 8, "SCALAR": scalar, "INITIALIZE": initialize}
    path.write_text(capture_ttir(indirect_recurrence,
        {"x": "*fp32", "indices": "*" + dtype, "out": "*fp32", "n": "i32", "m": "i32", "steps": "i32",
         **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("indirect_recurrence", (TensorBuffer("x", ("n",)), TensorBuffer("indices", ("n",), dtype=dtype),
        TensorBuffer("out", ("m",), "write" if fresh else "readwrite")), grid=(GridExtent("n", 1 if scalar else 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)), RuntimeScalar("m", "i32", 4, bounds=(0, 31)),
                         RuntimeScalar("steps", "i32", 5, bounds=(0, 5))))
    return convert_shared(path, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer"), abi


def arrays(indices, m, steps, *, fresh=False, dtype=np.int32):
    indices = np.asarray(indices, dtype=dtype)
    x, old = np.arange(len(indices), dtype=np.float32), np.arange(m, dtype=np.float32) + 100
    active = indices >= 0
    expected = np.empty_like(old) if fresh else old.copy()
    if fresh:
        expected[indices[active]] = x[active]
    for _ in range(steps):
        expected[indices[active]] = ((expected[indices[active]] + x[active]) + np.float32(1)) + np.float32(2)
    return [x, indices, *([] if fresh else [old]), *[np.asarray(n, dtype=np.int32) for n in (len(indices), m, steps)]], expected


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("dtype", ["i32", "i64"])
def test_gathers_observe_each_store_in_runtime_recurrences(tmp_path, scalar, fresh, dtype):
    from t2m.native import execute_native
    result, abi = recurrence(tmp_path, scalar=scalar, fresh=fresh, initialize=fresh, dtype=dtype)
    indices = [7, 0, 6, 2, 5, 1, 4, 3, -2147483648]
    values, expected = arrays(indices, 8, 3, fresh=fresh, dtype={"i32": np.int32, "i64": np.int64}[dtype])
    (actual,), _ = execute_native(result.mlir_text, values, tmp_path / "native", python=NATIVE, shape_abi=abi.runtime_shapes())
    np.testing.assert_array_equal(actual, expected)
    admission = json.loads(result.auxiliary_artifacts["memory-admission.json"])
    assert admission["schema"] == "t2m.memory-admission.v2" and admission["read_adaptations"]
    assert all(a["source_ids"] for a in admission["read_adaptations"])
    assert admission["guard_storage_bytes_per_output_element"] == 16
    assert "scf.forall" in result.mlir_text and "memref.load" in result.mlir_text


@pytest.mark.parametrize("scalar", [False, True])
def test_fresh_reads_refuse_before_any_completed_write(tmp_path, scalar):
    from t2m.native import execute_native
    result, abi = recurrence(tmp_path, scalar=scalar, fresh=True)
    values, _ = arrays([1, 0], 2, 1, fresh=True)
    with pytest.raises(ValueError, match=r"native host execution failed \(exit -6\)"):
        execute_native(result.mlir_text, values, tmp_path / "native", python=NATIVE, shape_abi=abi.runtime_shapes())


def sharing(root, write=False, writer=0):
    from t2m.examples.tensor_kernels import shared_writable_reads
    from t2m.frontend import capture_ttir
    constants = {"N": 13, "B": 8, "WRITE_SHARED": write, "WRITER": writer}
    path = root / "source.ttir"
    path.write_text(capture_ttir(shared_writable_reads,
        {"state": "*fp32", "indices": "*i32", "out": "*fp32", **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI("shared_writable_reads", (TensorBuffer("state", (4,), "readwrite"),
        TensorBuffer("indices", (13,), dtype="i32"), TensorBuffer("out", (13,), "write")), grid=(2,))
    return convert_shared(path, abi, root / "translation", optimizer=OPT, memory_profile="guarded-buffer")


def test_multiple_programs_can_read_shared_initial_state_without_writing_it(tmp_path):
    from t2m.native import execute_native
    result = sharing(tmp_path)
    state, indices = np.arange(4, dtype=np.float32) + 10, np.zeros(13, dtype=np.int32)
    (actual_state, actual), _ = execute_native(result.mlir_text, [state, indices], tmp_path / "native", python=NATIVE)
    np.testing.assert_array_equal(actual_state, state)
    np.testing.assert_array_equal(actual, np.full(13, 11, dtype=np.float32))


@pytest.mark.parametrize("writer", [0, 1])
def test_read_write_conflicts_refuse_in_both_observation_orders(tmp_path, writer):
    from t2m.native import execute_native
    result = sharing(tmp_path, write=True, writer=writer)
    with pytest.raises(ValueError, match=r"native host execution failed \(exit -6\)"):
        execute_native(result.mlir_text, [np.ones(4, dtype=np.float32), np.zeros(13, dtype=np.int32)], tmp_path / "native", python=NATIVE)


def test_structured_mutable_loads_keep_their_independent_copy(tmp_path):
    from t2m.cli import shared_demo
    from t2m.bundle import verify_bundle
    capture = tmp_path / "capture"
    shared_demo(capture, kernel="runtime-memory-store", optimizer=OPT, python=NATIVE, memory_profile="guarded-buffer")
    verify_bundle(capture)
    admission = json.loads((capture / "memory-admission.json").read_text())
    assert any(a["buffer"] == "out" and a["effect"] == "read" for a in admission["accesses"])


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("scalar", [False, True])
def test_one_merlin_library_preserves_mutable_observations_across_invocations(tmp_path, scalar):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result, _ = recurrence(tmp_path, scalar=scalar)
    values, expected = arrays([7, 0, 6, -1, 5, 1, 4, 3, -99], 11, 3)
    capture, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, capture, values, [expected], work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(capture, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for ordinal, (indices, m, steps) in enumerate((([], 0, 0), ([-1], 0, 3), ([3, 0], 4, 0),
            ([3, 0], 4, 1), ([11, 2, 9, 0, -1, 8, 6, 1, 7], 13, 5))):
        values, expected = arrays(indices, m, steps)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(values)})
        np.savez(reference, out0=expected)
        out = tmp_path / ("invoke" + str(ordinal))
        report = invoke_kernel(compiled, inputs, reference, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library
