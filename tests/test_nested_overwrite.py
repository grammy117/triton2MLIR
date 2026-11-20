"""Cartesian source loop coverage must prove every fresh tensor element."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not os.environ.get("TRITON_SHARED_OPT"), reason="existing shared optimizer required")
CONFIGURATIONS = ((False, 0, False, "full"), (False, 1, False, "full"), (True, 0, False, "full"),
                  (True, 1, False, "full"), (True, 0, True, "full"), (True, 1, True, "full"),
                  (False, 0, False, "branch-complete"))


def capture(root, order=0, mode="full", volume=False, unit_axis=False):
    from t2m.examples.nested_overwrite_kernels import (
        branch_overwrite,
        nested_overwrite,
        nested_volume_overwrite,
    )

    kernel = nested_volume_overwrite if volume else nested_overwrite
    if mode == "branch-complete":
        kernel = branch_overwrite
    dimensions, maxima = (("a", "b", "c"), (4, 5, 1 if unit_axis else 7)) if volume else (("rows", "cols"), (7, 9))
    types = {**{name: "*i64" for name in ("x", "bias", "out", "seen")}, **{name: "i32" for name in dimensions},
             "ORDER": "constexpr"}
    constants = {"ORDER": order}
    scalar = [RuntimeScalar(name, "i32", 4 + i, bounds=(0, maximum)) for i, (name, maximum) in enumerate(zip(dimensions, maxima, strict=True))]
    if not volume:
        types.update(count="i32", enabled="i32", MODE="constexpr")
        constants["MODE"] = mode
        scalar.extend((RuntimeScalar("count", "i32", 6, bounds=(0, 7)), RuntimeScalar("enabled", "i32", 7, bounds=(0, 1))))
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, types, constants))
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", dimensions, "read", "i64"),
        TensorBuffer("bias", (dimensions[0] if order == 0 else dimensions[-1],), "read", "i64"),
        TensorBuffer("out", dimensions, "write", "i64"), TensorBuffer("seen", (2,), "write", "i64")),
        runtime_scalars=tuple(scalar))
    return source, abi


@pytest.mark.parametrize("volume,order,unit,mode", CONFIGURATIONS)
def test_original_nested_traversal_can_initialize_complete_tensor_results(tmp_path, volume, order, unit, mode):
    source, abi = capture(tmp_path, order, mode, volume=volume, unit_axis=unit)
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    assert result.trace["memory"]["output_initialization"]["out"]["initial_value_required"] is False


@pytest.mark.parametrize("mode", ("different", "step", "dependent", "duplicate", "branch", "masked", "before-read"))
def test_partial_conditional_duplicate_or_uninitialized_traversals_cannot_gain_complete_credit(tmp_path, mode):
    source, abi = capture(tmp_path, mode=mode)
    with pytest.raises(ConversionError):
        convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def test_rectangular_coverage_gives_no_parallel_write_ownership(tmp_path):
    source, abi = capture(tmp_path)
    with pytest.raises(ConversionError):
        convert_shared(source, replace(abi, grid=(2,)), tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def arguments(volume, order, unit, index):
    shapes = ((0, 3, 1), (2, 0, 1), (1, 1, 1), (3, 4, 1 if unit else 5), (4, 5, 1 if unit else 7)) if volume else (
        (0, 7), (3, 0), (1, 1), (4, 7), (7, 9))
    shape = shapes[index]
    x = np.arange(np.prod(shape), dtype=np.int64).reshape(shape) * 37 + (1 << 55)
    axis = 0 if order == 0 else len(shape) - 1
    bias = np.arange(shape[axis], dtype=np.int64) * 19 + (1 << 54) + 1
    broadcast = [1] * len(shape)
    broadcast[axis] = len(bias)
    expected = x ^ bias.reshape(broadcast)
    seen = np.asarray([expected.flat[0], expected.flat[-1]] if expected.size else [-19, -23], np.int64)
    controls = [np.asarray(value, np.int32) for value in shape]
    if not volume:
        controls.extend((np.asarray(shape[0], np.int32), np.asarray(index % 2, np.int32)))
    return [x, bias, *controls], [expected, seen]


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native tools required")
@pytest.mark.parametrize("volume,order,unit,mode", CONFIGURATIONS)
def test_native_nests_keep_original_traversal_reuse_empty_axes_and_observations(tmp_path, volume, order, unit, mode):
    from t2m.native import execute_native

    source, abi = capture(tmp_path, order, mode, volume=volume, unit_axis=unit)
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    for index in range(5):
        inputs, expected = arguments(volume, order, unit, index)
        actual, _ = execute_native(result.mlir_text, inputs, tmp_path / f"native-{index}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=abi.runtime_shapes())
        for observed, wanted in zip(actual, expected, strict=True):
            assert (observed.shape, observed.dtype, observed.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native tools required")
@pytest.mark.parametrize("effect", ("read", "write", "readwrite"))
@pytest.mark.parametrize("shape", ((0,), (2, 0, 3)))
def test_empty_fresh_storage_needs_no_writer_but_active_reads_still_abort(tmp_path, effect, shape):
    from t2m.examples.nested_overwrite_kernels import empty_result
    from t2m.native import execute_native

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(empty_result, {"out": "*i64", "seen": "*i64", "enabled": "i32"}, {}))
    abi = TensorABI("empty_result", (TensorBuffer("out", shape, effect, "i64"),
        TensorBuffer("seen", (1,), "write", "i64")), runtime_scalars=(RuntimeScalar("enabled", "i32", 2, bounds=(0, 1)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    prefix = [] if effect == "write" else [np.empty(shape, np.int64)]
    actual, _ = execute_native(result.mlir_text, [*prefix, np.asarray(0, np.int32)], tmp_path / "native-empty",
        python=os.environ["T2M_NATIVE_PYTHON"])
    assert actual[-1].tolist() == [7]
    if effect != "read":
        assert actual[0].shape == shape
    if effect == "write":
        assert result.trace["memory"]["output_initialization"]["out"]["coverage"] == "zero-element-result"
    out = tmp_path / "aborted"
    with pytest.raises(ValueError, match="native host execution failed") as failure:
        execute_native(result.mlir_text, [*prefix, np.asarray(1, np.int32)], out, python=os.environ["T2M_NATIVE_PYTHON"])
    assert "exit -6" in str(failure.value) and not (out / "outputs.npz").exists()
    (tmp_path / "empty-active-abort.json").write_text(json.dumps({"error": str(failure.value)}) + "\n")


@pytest.fixture(scope="module", params=CONFIGURATIONS)
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(name) for name in ("T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing native/Merlin tools required")
    volume, order, unit, mode = request.param
    root = tmp_path_factory.mktemp(f"nested-{volume}-{order}-{unit}-{mode}")
    source, abi = capture(root, order, mode, volume=volume, unit_axis=unit)
    result = convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    values, expected = arguments(volume, order, unit, 3)
    write_shared_bundle(result, root / "capture", values, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, volume, order, unit


def invocation(root, volume, order, unit, index, name):
    values, expected = arguments(volume, order, unit, index)
    inputs, reference = root / f"inputs-{name}.npz", root / f"reference-{name}.npz"
    np.savez(inputs, **{f"in{i}": value for i, value in enumerate(values)})
    np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
    return inputs, reference


def test_one_merlin_library_retains_empty_axes_permuted_traversal_and_both_branches(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import verify_kernel_qualification

    root, volume, order, unit = compiled
    library, receipt = _record(root / "merlin/host/model.so"), verify_kernel_qualification(root / "merlin")
    for index in range(5):
        inputs, reference = invocation(root, volume, order, unit, index, str(index))
        out = root / f"invoke-{index}"
        report = invoke_kernel(root / "merlin", inputs, reference, out,
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(out)
    assert _record(root / "merlin/host/model.so") == library and verify_kernel_qualification(root / "merlin") == receipt


def test_actual_serial_images_preserve_source_nests_and_all_results(compiled):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    root, volume, order, unit = compiled
    for index in (0, 3):
        inputs, reference = invocation(root, volume, order, unit, index, "target-" + str(index))
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=inputs, reference=reference, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report


@pytest.mark.parametrize("effect", ("write", "readwrite"))
def test_static_empty_tensor_ports_execute_through_merlin_and_actual_serial_image(tmp_path, effect):
    from test_spike_model import TOOLS, Q

    from t2m.examples.nested_overwrite_kernels import empty_result
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not TOOLS or not os.environ.get("T2M_NATIVE_PYTHON"):
        pytest.skip("existing native/Merlin/serial tools required")
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(empty_result, {"out": "*i64", "seen": "*i64", "enabled": "i32"}, {}))
    abi = TensorABI("empty_result", (TensorBuffer("out", (2, 0, 3), effect, "i64"), TensorBuffer("seen", (1,), "write", "i64")),
        runtime_scalars=(RuntimeScalar("enabled", "i32", 2, bounds=(0, 1)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    values = [np.asarray(0, np.int32)] if effect == "write" else [np.empty((2, 0, 3), np.int64), np.asarray(0, np.int32)]
    write_shared_bundle(result, tmp_path / "capture", values, [np.empty((2, 0, 3), np.int64), np.asarray([7], np.int64)],
        work=tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(tmp_path / "capture", tmp_path / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    report = Q.qualify(tmp_path / "capture", tmp_path / "target", comparison="words",
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
    assert report["evaluation"]["result_count"] == 2
    assert Q.verify(tmp_path / "target", merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
