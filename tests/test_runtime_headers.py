"""Variable source headers preserve tiled reads, numeric state and zero paths."""

import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/native tools required")
KINDS = ("rows", "columns", "tile")


def capture(root, kind, dtype="i32", profile="tensor", *, bounds=None, missing_zero=False):
    from t2m.examples.runtime_header_kernels import runtime_partition, selected_row_sum

    if kind == "tile":
        kernel = runtime_partition
        names = ("x", "bias", "out", "seen")
        constants = {"B": 8, "MISSING_ZERO": missing_zero}
        buffers = (TensorBuffer("x", ("n",), "read", "i64"), TensorBuffer("bias", (1,), "read", "i64"),
                   TensorBuffer("out", ("n",), "write", "i64"), TensorBuffer("seen", ("n",), "write", "i64"))
        scalars = (RuntimeScalar("n", dtype, 4, bounds=(0, 31)),)
        positions, grid = (5, 6, 7), (GridExtent("n", 8),)
        control_bounds = ((-2, 3), (-2, 8), (1, 3))
    else:
        kernel = selected_row_sum
        names = ("x", "out")
        constants = {"N": 17, "R": 8, "B": 8, "TRANSPOSE": kind == "columns"}
        buffers = (TensorBuffer("x", (17, 8) if kind == "columns" else (8, 17), "read", "i64"),
                   TensorBuffer("out", (17,), "write", "i64"))
        scalars, positions, grid = (), (2, 3, 4), (3,)
        control_bounds = ((0, 3), (0, 8), (1, 3))
    control_bounds = control_bounds if bounds is None else bounds
    scalars += tuple(RuntimeScalar(name, dtype, position, bounds=interval) for name, position, interval in
                     zip(("lower", "upper", "stride"), positions, control_bounds, strict=True))
    signature = {**{name: "*i64" for name in names}, **{scalar.name: dtype for scalar in scalars},
                 **dict.fromkeys(constants, "constexpr")}
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, signature, constants))
    abi = TensorABI(kernel.__name__, buffers, grid=grid, runtime_scalars=scalars)
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile=profile)


def tile_reference(x, bias, lower, upper, stride):
    word = x ^ bias[0]
    positive, state, seen = word > 0, word ^ 1, word ^ 64
    for iteration in range(lower, upper, stride):
        earlier = np.where(positive, state, -17)
        state = np.where(positive, state, word ^ np.int64(iteration + 2) ^ earlier)
        seen ^= state
        state = np.where(positive, state ^ 8, state)
    if lower >= upper:
        state = np.where(positive, state, word ^ 16)
    return [state, seen]


def arguments(kind, dtype, index):
    lower, upper, stride = ((0, 8, 1), (1, 8, 3), (3, 1, 2), (0, 0, 1), (2, 7, 2), (3, 8, 1))[index]
    integer = np.int32 if dtype == "i32" else np.int64
    if kind == "tile":
        if index == 4:
            lower, upper, stride = -2, 3, 2
        n = (0, 1, 8, 17, 31, 9)[index]
        x = np.arange(n, dtype=np.int64) * 37 + (1 << 56)
        x[1::2] *= -1
        if n:
            x[-1] = (0, (1 << 63)-1, -(1 << 63), -1, 0, -65535)[index]
        bias = np.asarray([(1 << 55) + index], np.int64)
        arrays = [x, bias, np.asarray(n, integer)]
        expected = tile_reference(x, bias, lower, upper, stride)
    else:
        x = np.arange(8 * 17, dtype=np.int64).reshape(8, 17) * 37 + (1 << 55)
        x[1::2] *= -1
        x.flat[0], x.flat[-1] = -(1 << 63), (1 << 63)-1
        expected = [np.sum(x[lower:upper:stride], axis=0, dtype=np.int64)]
        arrays = [x.T.copy() if kind == "columns" else x]
    return [*arrays, *[np.asarray(value, integer) for value in (lower, upper, stride)]], expected


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", ("i32", "i64"))
@pytest.mark.parametrize("profile", ("tensor", "buffer"))
def test_original_ssa_headers_and_tile_axes_survive_conversion(tmp_path, kind, dtype, profile):
    result = capture(tmp_path, kind, dtype, profile)
    assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    if profile == "tensor":
        assert not any(op.name.startswith("memref.") for op in result.module.walk())
        loop = next(op for op in result.module.walk() if op.name == "scf.for")
        entry = next(iter(result.module.ops))
        assert tuple(loop.operands[:3]) == tuple(entry.body.block.args[-3:])
    if kind != "tile":
        proof = result.trace["memory"]["source_projected_reads"][0]
        assert proof["retained_axes"] == ([0] if kind == "columns" else [1])
        assert proof["source_tile"] == [8]
        assert result.trace["memory"]["source_loops"][0]["induction"] == "source-IV-enclosure; not-an-iteration-ordinal"


@pytest.mark.parametrize("bounds", [((0, 3), (0, 9), (1, 3)), ((0, 3), (0, 8), (0, 3)),
                                   ((0, 3), (0, 8), (1, (1 << 31)-1)), ((0, 3), None, (1, 3))])
def test_unproved_source_domains_invalid_steps_and_active_axis_overflow_still_refuse(tmp_path, bounds):
    with pytest.raises(ConversionError):
        capture(tmp_path, "rows", bounds=bounds)


def test_a_zero_trip_source_path_cannot_borrow_body_initialization(tmp_path):
    with pytest.raises(ConversionError):
        capture(tmp_path, "tile", missing_zero=True)


@pytest.mark.parametrize("name,value", [("lower", -1), ("upper", 9), ("stride", 0)])
def test_runtime_header_premises_are_actual_compiled_entry_guards(tmp_path, name, value):
    from t2m.native import observe_native

    result = capture(tmp_path, "rows")
    inputs, _ = arguments("rows", "i32", 1)
    inputs[{"lower": 1, "upper": 2, "stride": 3}[name]] = np.asarray(value, np.int32)
    # Static tensor ports need no shape preflight. The invalid scalar reaches
    # the original emitted assertion in actual compiled execution.
    assert result.spec.runtime_shapes() is None
    out = tmp_path / "invalid-header"
    report = observe_native(result.mlir_text, inputs, out, python=os.environ["T2M_NATIVE_PYTHON"])
    assert report["returncode"] == -6 and not (out / "outputs.npz").exists()
    assert (out / "invocation-started.json").exists()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", ("i32", "i64"))
@pytest.mark.parametrize("profile", ("tensor", "buffer"))
def test_native_source_recurrences_tail_masks_and_empty_paths(tmp_path, kind, dtype, profile):
    from t2m.native import execute_native

    result = capture(tmp_path, kind, dtype, profile)
    for index in range(6):
        inputs, expected = arguments(kind, dtype, index)
        actual, _ = execute_native(result.mlir_text, inputs, tmp_path / f"native-{index}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())


@pytest.fixture(scope="module", params=[(kind, dtype) for kind in KINDS for dtype in ("i32", "i64")])
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(name) for name in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing Merlin tools required")
    kind, dtype = request.param
    root = tmp_path_factory.mktemp("runtime-header-" + kind + "-" + dtype)
    result = capture(root, kind, dtype)
    inputs, expected = arguments(kind, dtype, 1)
    write_shared_bundle(result, root / "capture", inputs, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, kind, dtype


def test_one_merlin_library_retains_every_actual_runtime_header(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    root, kind, dtype = compiled
    library = _record(root / "merlin/host/model.so")
    for index in range(6):
        inputs, expected = arguments(kind, dtype, index)
        arguments_file, reference_file = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
        np.savez(arguments_file, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference_file, **{f"out{i}": value for i, value in enumerate(expected)})
        report = invoke_kernel(root / "merlin", arguments_file, reference_file, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(root / f"invoke-{index}")
    assert _record(root / "merlin/host/model.so") == library


def test_actual_serial_images_preserve_strided_and_empty_source_paths(compiled):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    root, kind, dtype = compiled
    for index in (1, 2):
        inputs, expected = arguments(kind, dtype, index)
        arguments_file, reference_file = root / f"target-arguments-{index}.npz", root / f"target-reference-{index}.npz"
        np.savez(arguments_file, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference_file, **{f"out{i}": value for i, value in enumerate(expected)})
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=arguments_file, reference=reference_file, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == len(expected)
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
