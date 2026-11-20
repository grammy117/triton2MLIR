"""Actual Boolean alternatives cover local reads and complete result writes."""

import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing shared/native tools required")
MODES = ("branch", "masks", "nested-branch", "nested", "select", "overlap", "observed", "rows", "columns", "float")


def capture(root, mode, *, grid=(1,)):
    from t2m.examples.partition_kernels import (
        float_partition,
        partition_rows,
        partition_transform,
    )

    source = root / "source.ttir"
    if mode == "float":
        source.write_text(capture_ttir(float_partition, {"x": "*fp32", "out": "*i64", "seen": "*i64", "n": "i32"}, {}))
        abi = TensorABI("float_partition", (TensorBuffer("x", ("n",), "read", "f32"),
            TensorBuffer("out", ("n",), "write", "i64"), TensorBuffer("seen", ("n",), "write", "i64")), grid=grid,
            runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)),))
        return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    if mode in ("rows", "columns"):
        source.write_text(capture_ttir(partition_rows,
            {**{name: "*i64" for name in ("x", "bias", "out", "seen")}, "rows": "i32", "cols": "i32",
             "control": "i32", "ORDER": "constexpr"}, {"ORDER": int(mode == "columns")}))
        axis = "rows" if mode == "rows" else "cols"
        abi = TensorABI("partition_rows", (TensorBuffer("x", ("rows", "cols"), "read", "i64"),
            TensorBuffer("bias", (axis,), "read", "i64"), TensorBuffer("out", ("rows", "cols"), "write", "i64"),
            TensorBuffer("seen", (axis,), "write", "i64")), grid=grid,
            runtime_scalars=(RuntimeScalar("rows", "i32", 4, bounds=(0, 7)), RuntimeScalar("cols", "i32", 5, bounds=(0, 9)),
                             RuntimeScalar("control", "i32", 6, bounds=(-(1 << 31), (1 << 31)-1))))
        return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    source.write_text(capture_ttir(partition_transform,
        {**{name: "*i64" for name in ("x", "bias", "out", "seen")}, "n": "i32", "control": "i32", "MODE": "constexpr"},
        {"MODE": mode}))
    abi = TensorABI("partition_transform", (TensorBuffer("x", ("n",), "read", "i64"),
        TensorBuffer("bias", (1,), "read", "i64"), TensorBuffer("out", ("n",), "write", "i64"),
        TensorBuffer("seen", ("n",), "write", "i64")), grid=grid,
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
                         RuntimeScalar("control", "i32", 5, bounds=(-(1 << 31), (1 << 31)-1))))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def reference(x, bias, control, mode):
    value = x ^ bias[0]
    p, q = value > 0, control != 0
    if mode in ("nested", "nested-branch"):
        result = np.where(p & q, value ^ 1, np.where(p, value ^ 2, value ^ 4))
    elif mode == "select":
        result = np.where(p if q else ~p, value ^ 1, value ^ 2)
    elif mode == "overlap":
        result = np.where(~p | (not q), value ^ 2, value ^ 1)
    else:
        result = np.where(p, value ^ 1, value ^ 2)
    seen = np.where(p, value ^ 1, -17) if mode == "observed" else result.copy()
    return [result, seen]


def arguments(mode, index):
    control = (0, -1, -(1 << 31), (1 << 31)-1, 0)[index]
    if mode == "float":
        n = (0, 1, 3, 17, 31)[index]
        bits = np.resize(np.asarray([0, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
            0x7fc00011, 0xffc01234, 0x7f800001, 0xff800001, 1, 0x80000001, 0x7f7fffff, 0xff7fffff], np.uint32), n)
        magnitude = bits & 0x7fffffff
        # Independent IEEE word classification: NaNs are unordered, not <=0.
        positive = ((bits >> 31) == 0) & (magnitude > 0) & (magnitude <= 0x7f800000)
        expected = np.where(positive, 11, -23).astype(np.int64)
        return [bits.view(np.float32), np.asarray(n, np.int32)], [expected, expected.copy()]
    if mode in ("rows", "columns"):
        shape = ((0, 7), (3, 0), (1, 1), (4, 7), (7, 9))[index]
        x = np.arange(np.prod(shape), dtype=np.int64).reshape(shape) * 37 + (1 << 55)
        x.flat[1::2] *= -1
        axis = 0 if mode == "rows" else 1
        bias = np.arange(shape[axis], dtype=np.int64) * 19 + (1 << 54) + index
        value = x ^ bias.reshape((-1, 1) if axis == 0 else (1, -1))
        expected = np.where(value > 0, value ^ (1 if control else 8), value ^ (2 if control else 4))
        planes = expected if axis == 0 else expected.T
        seen = planes[:, 0].copy() if shape[1-axis] else np.full(shape[axis], -17, np.int64)
        return [x, bias, *(np.asarray(n, np.int32) for n in shape), np.asarray(control, np.int32)], [expected, seen]
    n = (0, 1, 3, 17, 31)[index]
    x = np.arange(n, dtype=np.int64) * 37 + (1 << 55)
    x[1::2] *= -1
    if n:
        x[-1] = (0, (1 << 63)-1, -(1 << 63), -1, 0)[index]
    bias = np.asarray([(1 << 54) + index], np.int64)
    return [x, bias, np.asarray(n, np.int32), np.asarray(control, np.int32)], reference(x, bias, control, mode)


@pytest.mark.parametrize("mode", MODES)
def test_actual_source_partitions_retain_all_paths_and_functional_outputs(tmp_path, mode):
    result = capture(tmp_path, mode)
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    proofs = [row for loop in result.trace["memory"]["source_loops"]
              for row in loop.get("complete_scalar_result_proofs", {}).values()]
    assert any(row.get("condition_coverage") == "actual-source-boolean-partition" for row in proofs)
    if mode == "observed":
        assert any(row.get("initialization", {}).get("condition_coverage") == "actual-source-boolean-implication"
                   for row in result.trace["memory"]["source_gathers"])


@pytest.mark.parametrize("mode", ("missing", "duplicate", "independent", "different-index", "early", "nested-missing"))
def test_holes_unrelated_masks_wrong_indices_and_early_reads_remain_located_refusals(tmp_path, mode):
    with pytest.raises(ConversionError) as failure:
        capture(tmp_path, mode)
    assert failure.value.source_id is not None


def test_boolean_coverage_supplies_no_parallel_write_ownership(tmp_path):
    with pytest.raises(ConversionError):
        capture(tmp_path, "masks", grid=(2,))


@pytest.mark.parametrize("bounds", ((0, 0), (0, 1)))
def test_zero_trip_body_writes_cannot_initialize_an_observed_element(tmp_path, bounds):
    from t2m.examples.partition_kernels import zero_trip_write

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(zero_trip_write, {"out": "*i64", "n": "i32"}, {}))
    abi = TensorABI("zero_trip_write", (TensorBuffer("out", (2,), "write", "i64"),),
        runtime_scalars=(RuntimeScalar("n", "i32", 1, bounds=bounds),))
    with pytest.raises(ConversionError):
        convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def test_ordered_float_comparisons_cannot_claim_coverage_of_nan_inputs(tmp_path):
    from t2m.examples.partition_kernels import ordered_float_gap

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(ordered_float_gap, {"x": "*fp32", "out": "*i64", "seen": "*i64", "n": "i32"}, {}))
    abi = TensorABI("ordered_float_gap", (TensorBuffer("x", ("n",), "read", "f32"),
        TensorBuffer("out", ("n",), "write", "i64"), TensorBuffer("seen", ("n",), "write", "i64")),
        runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 31)),))
    with pytest.raises(ConversionError) as failure:
        convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    assert failure.value.source_id is not None


@pytest.mark.parametrize("mode", MODES)
def test_native_all_alternatives_keep_signed_words_masks_fills_and_order(tmp_path, mode):
    from t2m.native import execute_native

    result = capture(tmp_path, mode)
    for index in range(5):
        inputs, expected = arguments(mode, index)
        actual, _report = execute_native(result.mlir_text, inputs, tmp_path / f"native-{index}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())


@pytest.fixture(scope="module", params=MODES)
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing Merlin tools required")
    mode = request.param
    root = tmp_path_factory.mktemp("partition-" + mode)
    result = capture(root, mode)
    inputs, expected = arguments(mode, 3)
    write_shared_bundle(result, root / "capture", inputs, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, mode


def invocation(root, mode, index, name):
    inputs, expected = arguments(mode, index)
    arguments_file, reference_file = root / f"inputs-{name}.npz", root / f"reference-{name}.npz"
    np.savez(arguments_file, **{f"in{i}": value for i, value in enumerate(inputs)})
    np.savez(reference_file, **{f"out{i}": value for i, value in enumerate(expected)})
    return arguments_file, reference_file


def test_one_merlin_library_preserves_each_partition_and_full_control_words(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import verify_kernel_qualification

    root, mode = compiled
    library, receipt = _record(root / "merlin/host/model.so"), verify_kernel_qualification(root / "merlin")
    for index in range(5):
        inputs, reference_file = invocation(root, mode, index, str(index))
        out = root / f"invoke-{index}"
        report = invoke_kernel(root / "merlin", inputs, reference_file, out,
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(out)
    assert _record(root / "merlin/host/model.so") == library and verify_kernel_qualification(root / "merlin") == receipt


def test_actual_serial_images_retain_source_choices_order_and_all_observations(compiled):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    root, mode = compiled
    for index in (0, 3):
        inputs, reference_file = invocation(root, mode, index, "target-" + str(index))
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=inputs, reference=reference_file, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
