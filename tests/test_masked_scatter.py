"""Nonrectangular masks preserve state inside proved independent tiles."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing native/shared environments required")


def capture(root, *, mode=0, fresh=False, step=8, declared_fresh=None):
    from t2m.examples.tensor_kernels import masked_tile_store
    from t2m.frontend import capture_ttir
    path = root / "source.ttir"
    path.write_text(capture_ttir(masked_tile_store,
        {"x": "*fp32", "out": "*fp32", "n": "i32", "B": "constexpr", "STEP": "constexpr", "MODE": "constexpr", "FRESH": "constexpr"},
        {"B": 8, "STEP": step, "MODE": mode, "FRESH": fresh}))
    if declared_fresh is None:
        declared_fresh = fresh
    abi = TensorABI("masked_tile_store", (TensorBuffer("x", ("n",)),
        TensorBuffer("out", ("n",), "write" if declared_fresh else "readwrite")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 13)),))
    return path, abi


def arguments(n, mode, fresh):
    x = np.linspace(-2, 2, n, dtype=np.float32)
    old = np.arange(n, dtype=np.float32) + 100
    if n > 1:
        old.view(np.uint32)[1] = 0x7FC12345
        old.view(np.uint32)[0] = 0x80000000
    index = np.arange(n)
    periodic, data = index % 3 != 1, x > 0
    selected = periodic if mode == 0 else data if mode == 1 else periodic | data
    expected = x.copy() if fresh else old.copy()
    for update in range(2):
        expected[selected] = (x[selected] + np.float32(update)) + np.float32(1)
    return ([x, *([] if fresh else [old]), np.asarray(n, dtype=np.int32)], expected, selected, old)


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("fresh", [False, True])
def test_periodic_data_and_boolean_masks_retain_source_loops(tmp_path, profile, mode, fresh):
    from t2m.native import execute_native
    from t2m.shared_tensor import convert_shared
    path, abi = capture(tmp_path, mode=mode, fresh=fresh)
    result = convert_shared(path, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    arrays, expected, selected, old = arguments(13, mode, fresh)
    (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / "native", python=NATIVE, shape_abi=abi.runtime_shapes())
    np.testing.assert_array_equal(actual, expected)
    if not fresh:
        np.testing.assert_array_equal(actual.view(np.uint32)[~selected], old.view(np.uint32)[~selected])
    assert result.trace["memory"]["source_scatters"]
    assert any(op.name == "scf.for" for op in result.module.walk())
    if profile == "tensor":
        assert any(op.name == "arith.select" for op in result.module.walk())
    else:
        assert "memref.store" in result.mlir_text and "scf.if" in result.mlir_text


@pytest.mark.parametrize("change,reason", [("fresh", "read before"), ("tail", "mask does not prove"),
                                          ("overlap", "parallel write ownership")])
def test_unproved_initialization_bounds_and_conflicts_refuse(tmp_path, change, reason):
    from t2m.diagnostics import ConversionError
    from t2m.shared_tensor import convert_shared
    path, abi = capture(tmp_path, mode=3 if change == "tail" else 0, step=0 if change == "overlap" else 8,
                        declared_fresh=change == "fresh")
    with pytest.raises(ConversionError, match=reason):
        convert_shared(path, abi, tmp_path / "translation", optimizer=OPT, memory_profile="buffer")
    assert not (tmp_path / "translation/memory-ir").exists()


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("profile", ["tensor", "buffer"])
@pytest.mark.parametrize("fresh", [False, True])
def test_one_merlin_library_keeps_masked_updates_for_changing_runtime_shapes(tmp_path, profile, fresh):
    from t2m.bundle import _record, verify_bundle
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    from t2m.shared_tensor import convert_shared
    path, abi = capture(tmp_path, mode=2, fresh=fresh)
    result = convert_shared(path, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    arrays, expected, _, _ = arguments(7, 2, fresh)
    capture_dir, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, capture_dir, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    verify_bundle(capture_dir)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(capture_dir, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for n in (0, 1, 7, 8, 9, 13):
        arrays, expected, selected, old = arguments(n, 2, fresh)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        out = tmp_path / ("invoke" + str(n))
        report = invoke_kernel(compiled, inputs, reference, out, merlin_root=root, python=python)
        verify_invocation(out)
        assert report["native_compilation"] == "reused-qualified-library"
        assert report["evaluation"]["functional_inputs_preserved"]
        if not fresh:
            with np.load(out / "evaluation/outputs.npz", allow_pickle=False) as values:
                np.testing.assert_array_equal(values["out0"].view(np.uint32)[~selected], old.view(np.uint32)[~selected])
    assert verify_kernel_qualification(compiled) == receipt
    assert _record(compiled / "host/model.so") == library


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_constant_offsets_are_clamped_for_empty_runtime_storage(tmp_path, profile):
    from t2m.examples.tensor_kernels import masked_offset_store
    from t2m.frontend import capture_ttir
    from t2m.native import execute_native
    from t2m.shared_tensor import convert_shared
    path = tmp_path / "source.ttir"
    path.write_text(capture_ttir(masked_offset_store, {"out": "*fp32", "n": "i32", "B": "constexpr"}, {"B": 8}))
    abi = TensorABI("masked_offset_store", (TensorBuffer("out", ("n",), "readwrite"),),
                    runtime_scalars=(RuntimeScalar("n", "i32", 1, bounds=(0, 13)),))
    result = convert_shared(path, abi, tmp_path / "translation", optimizer=OPT, memory_profile=profile)
    for n in (0, 1, 4, 7, 13):
        old = np.arange(n, dtype=np.float32)
        expected = old.copy()
        selected = np.arange(4, min(n, 12))
        selected = selected[selected % 3 != 1]
        expected[selected] = 7
        (actual,), _ = execute_native(result.mlir_text, [old, np.asarray(n, dtype=np.int32)], tmp_path / ("native" + str(n)),
                                     python=NATIVE, shape_abi=abi.runtime_shapes())
        np.testing.assert_array_equal(actual, expected)
