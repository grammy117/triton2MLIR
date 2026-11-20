"""Actual partial source writes support precisely their completed scalar reads."""

import os
from pathlib import Path

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing shared/native tools required")
CONFIGURATIONS = ((False, 0, "corners"), (False, 1, "corners"), (False, 0, "point"), (False, 1, "point"),
                  (False, 0, "reread"), (False, 1, "reread"), (False, 0, "and-mask"), (False, 1, "and-mask"),
                  (True, 0, "corners"), (True, 1, "corners"))


def capture(root, volume, order, mode, *, grid=(1,)):
    from t2m.examples.partial_memory_kernels import (
        progressive_transform,
        progressive_volume,
    )

    kernel = progressive_volume if volume else progressive_transform
    dimensions, maxima = (("a", "b", "c"), (4, 5, 7)) if volume else (("rows", "cols"), (7, 9))
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel,
        {**{name: "*i64" for name in ("x", "bias", "out", "first", "last")},
         **{name: "i32" for name in dimensions}, "ORDER": "constexpr", **({} if volume else {"MODE": "constexpr"})},
        {"ORDER": order, **({} if volume else {"MODE": mode})}))
    axis = dimensions[0] if order == 0 else dimensions[-1]
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", dimensions, "read", "i64"), TensorBuffer("bias", (axis,), "read", "i64"),
        TensorBuffer("out", dimensions, "write", "i64"), TensorBuffer("first", (axis,), "write", "i64"),
        TensorBuffer("last", (axis,), "write", "i64")), grid=grid, runtime_scalars=tuple(
            RuntimeScalar(name, "i32", 5 + i, bounds=(0, maximum)) for i, (name, maximum) in enumerate(zip(dimensions, maxima, strict=True))))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


@pytest.mark.parametrize("volume,order,mode", CONFIGURATIONS)
def test_completed_rows_columns_and_planes_supply_local_initialized_reads(tmp_path, volume, order, mode):
    result = capture(tmp_path, volume, order, mode)
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    reads = [row for row in result.trace["memory"]["source_gathers"] if row.get("initialization")]
    assert reads and all(row["initialization"]["coverage"] == "completed-source-store-image" for row in reads)


@pytest.mark.parametrize("mode", ("before-read", "masked", "future", "unmasked", "or-mask"))
def test_unwritten_future_masked_or_zero_trip_regions_do_not_gain_initialization(tmp_path, mode):
    with pytest.raises(ConversionError):
        capture(tmp_path, False, 0, mode)


def test_local_initialized_regions_do_not_grant_parallel_write_ownership(tmp_path):
    with pytest.raises(ConversionError):
        capture(tmp_path, False, 0, "corners", grid=(2,))


def arguments(volume, order, mode, index):
    shape = ((0, 3, 2), (2, 0, 3), (1, 1, 1), (3, 4, 5), (4, 5, 7))[index] if volume else (
        (0, 7), (3, 0), (1, 1), (4, 7), (7, 9))[index]
    x = np.arange(np.prod(shape), dtype=np.int64).reshape(shape) * 37 + (1 << 55)
    axis = 0 if order == 0 else len(shape) - 1
    bias = np.arange(shape[axis], dtype=np.int64) * 19 + (1 << 54) + 1
    broadcast = [1] * len(shape)
    broadcast[axis] = bias.size
    expected = x ^ bias.reshape(broadcast)
    if mode == "point":
        expected ^= 1
    planes = np.moveaxis(expected, axis, 0)
    inner_size = int(np.prod([n for i, n in enumerate(shape) if i != axis]))
    flat = planes.reshape(shape[axis], inner_size)
    first = (np.bitwise_xor.reduce(flat, axis=1) if mode == "reread" else flat[:, 0].copy()) if inner_size else np.full(shape[axis],
        0 if mode == "reread" else -19, np.int64)
    last = flat[:, -1].copy() if inner_size else np.full(shape[axis], -23, np.int64)
    if mode == "and-mask":
        first[np.arange(shape[axis]) % 2 == 1] = -19
    return [x, bias, *(np.asarray(n, np.int32) for n in shape)], [expected, first, last]


@pytest.mark.parametrize("volume,order,mode", CONFIGURATIONS)
def test_native_partial_reads_preserve_traversal_empty_axes_and_source_word_values(tmp_path, volume, order, mode):
    from t2m.native import execute_native

    result = capture(tmp_path, volume, order, mode)
    for index in range(5):
        values, expected = arguments(volume, order, mode, index)
        actual, _report = execute_native(result.mlir_text, values, tmp_path / f"native-{index}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())


@pytest.fixture(scope="module", params=CONFIGURATIONS)
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing Merlin tools required")
    volume, order, mode = request.param
    root = tmp_path_factory.mktemp(f"partial-{volume}-{order}-{mode}")
    result = capture(root, volume, order, mode)
    values, expected = arguments(volume, order, mode, 3)
    write_shared_bundle(result, root / "capture", values, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, volume, order, mode


def invocation(root, volume, order, mode, index, name):
    values, expected = arguments(volume, order, mode, index)
    inputs, reference = root / f"inputs-{name}.npz", root / f"reference-{name}.npz"
    np.savez(inputs, **{f"in{i}": value for i, value in enumerate(values)})
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
    return inputs, reference


def test_one_merlin_library_preserves_local_partial_reads_and_independent_shapes(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import verify_kernel_qualification

    root, volume, order, mode = compiled
    library, receipt = _record(root / "merlin/host/model.so"), verify_kernel_qualification(root / "merlin")
    for index in range(5):
        inputs, reference = invocation(root, volume, order, mode, index, str(index))
        out = root / f"invoke-{index}"
        report = invoke_kernel(root / "merlin", inputs, reference, out,
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(out)
    assert _record(root / "merlin/host/model.so") == library and verify_kernel_qualification(root / "merlin") == receipt


def test_actual_serial_images_preserve_partial_plane_point_and_recurrent_reads(compiled, monkeypatch):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    root, volume, order, mode = compiled
    for index in (0, 3):
        inputs, reference = invocation(root, volume, order, mode, index, "target-" + str(index))
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=inputs, reference=reference, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 3
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
