"""Scalar map branches retain zero-divisor guards, tuples and source loops."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
PROFILES = ["tensor", "buffer", "guarded-buffer"]


def capture(root, profile, dtype, while_loop):
    from t2m.examples.elementwise_map_kernels import elementwise_map_values
    from t2m.frontend import capture_ttir
    names = ("x", "y", "out_a", "out_b")
    source = root / "source.ttir"
    source.write_text(capture_ttir(elementwise_map_values, {**{name: "*fp32" if dtype == "f32" else "*i32" for name in names},
        "n": "i32", "iterations": "i32", "B": "constexpr", "WHILE_LOOP": "constexpr"}, {"B": 8, "WHILE_LOOP": while_loop}))
    abi = TensorABI("elementwise_map_values", tuple(TensorBuffer(name, ("n",), "write" if name.startswith("out") else "read", dtype=dtype)
        for name in names), grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
        RuntimeScalar("iterations", "i32", 5, bounds=(0, 4))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile=profile,
                          address_policy="source-words" if profile == "guarded-buffer" else "structured")


def arguments(n, count, dtype):
    numpy_dtype = np.float32 if dtype == "f32" else np.int32
    x = (np.arange(n) - 5).astype(numpy_dtype)
    y = np.resize(np.asarray([0, 2, -4, 0, -2, 4], dtype=numpy_dtype), n)
    first = np.full(n, -7, dtype=numpy_dtype)
    active = y != 0
    divided = x[active] / y[active]
    first[active] = divided if dtype == "f32" else np.trunc(divided).astype(np.int32)
    for step in range(count):
        first = first + y * numpy_dtype(step)
    return [x, y, np.asarray(n, dtype=np.int32), np.asarray(count, dtype=np.int32)], [first, x + y]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("dtype", ["f32", "i32"])
@pytest.mark.parametrize("while_loop", [False, True], ids=["for", "while"])
def test_scalar_map_keeps_branches_tuples_and_runtime_recurrences(tmp_path, profile, dtype, while_loop):
    from t2m.native import execute_native
    result = capture(tmp_path, profile, dtype, while_loop)
    assert any(t["kind"] == "ttir-elementwise-map" for t in result.trace["transformations"])
    assert any("prov.composite_family" in op.attributes for op in result.module.walk())
    for i, (n, count) in enumerate(((0, 0), (1, 4), (13, 0), (31, 3))):
        arrays, expected = arguments(n, count, dtype)
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                   shape_abi=result.spec.runtime_shapes())
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(value, reference)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("dtype", ["f32", "i32"])
@pytest.mark.parametrize("while_loop", [False, True], ids=["for", "while"])
def test_one_merlin_library_retains_scalar_map_control_and_inputs(tmp_path, profile, dtype, while_loop):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result = capture(tmp_path, profile, dtype, while_loop)
    arrays, expected = arguments(13, 3, dtype)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, expected, work=tmp_path / "native", python=NATIVE)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, (n, count) in enumerate(((0, 0), (1, 4), (13, 0), (31, 3))):
        arrays, expected = arguments(n, count, dtype)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, **{f"out{j}": value for j, value in enumerate(expected)})
        report = invoke_kernel(compiled, inputs, reference, tmp_path / f"invoke{i}", merlin_root=root, python=python)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library


@pytest.mark.parametrize("kind", ["packed", "memory"])
def test_map_requires_logical_pack_and_memory_contracts(tmp_path, kind):
    from t2m.diagnostics import ConversionError
    from t2m.examples.elementwise_map_kernels import packed_map_values, pointer_load_map
    from t2m.frontend import capture_ttir
    kernel = packed_map_values if kind == "packed" else pointer_load_map
    names = ("x", "y", "out") if kind == "packed" else ("x", "out")
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(kernel, {name: "*fp32" for name in names}, {}))
    abi = TensorABI(kernel.__name__, tuple(TensorBuffer(name, (8,), "write" if name == "out" else "read") for name in names))
    with pytest.raises(ConversionError, match="logical grouping" if kind == "packed" else "source memory") as exc:
        convert_shared(source, abi, tmp_path / "translation", optimizer=OPT,
                       memory_profile="guarded-buffer" if kind == "packed" else "tensor",
                       address_policy="source-words" if kind == "packed" else "structured")
    assert exc.value.opcode == "tt.map_elementwise" and exc.value.source_id
