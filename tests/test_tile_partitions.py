"""Real tiled masked alternatives retain parallel computation and observations."""

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
MODES = ("pair", "observed", "nested", "overlap", "float")


def capture(root, mode):
    from t2m.examples.tile_partition_kernels import tile_partition

    source = root / "source.ttir"
    source.write_text(capture_ttir(tile_partition,
        {"x": "*fp32" if mode == "float" else "*i64", **{name: "*i64" for name in ("bias", "out", "seen")},
         "n": "i32", "control": "i32", "B": "constexpr", "MODE": "constexpr"}, {"B": 8, "MODE": mode}))
    abi = TensorABI("tile_partition", (TensorBuffer("x", ("n",), "read", "f32" if mode == "float" else "i64"),
        TensorBuffer("bias", (1,), "read", "i64"), TensorBuffer("out", ("n",), "write", "i64"),
        TensorBuffer("seen", ("n",), "write", "i64")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
                         RuntimeScalar("control", "i32", 5, bounds=(-(1 << 31), (1 << 31)-1))))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def arguments(mode, index):
    n = (0, 1, 8, 17, 31)[index]
    control = (0, -(1 << 31), -1, (1 << 31)-1, 0)[index]
    bias = np.asarray([(1 << 55) + index], np.int64)
    if mode == "float":
        bits = np.resize(np.asarray([0, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
            0x7fc00011, 0xffc01234, 0x7f800001, 0xff800001, 1, 0x80000001, 0x7f7fffff, 0xff7fffff], np.uint32), n)
        magnitude = bits & 0x7fffffff
        p = ((bits >> 31) == 0) & (magnitude > 0) & (magnitude <= 0x7f800000)
        x = bits.view(np.float32)
        first, second = np.full(n, 11, np.int64), np.full(n, -23, np.int64)
    else:
        x = np.arange(n, dtype=np.int64) * 37 + (1 << 56)
        x[1::2] *= -1
        if n:
            x[-1] = (0, (1 << 63)-1, -(1 << 63), -1, 0)[index]
        p = x > 0
        value = x ^ bias[0]
        first, second = value ^ 1, value ^ 2
    q = control != 0
    if mode == "nested":
        result = np.where(p & q, first, np.where(p, second, value ^ 4))
    elif mode == "overlap":
        result = np.where(~p | (not q), second, first)
    else:
        result = np.where(p, first, second)
    seen = np.where(p, first, -17) if mode == "observed" else result.copy()
    return [x, bias, np.asarray(n, np.int32), np.asarray(control, np.int32)], [result, seen]


@pytest.mark.parametrize("mode", MODES)
def test_fresh_tiled_partitions_remain_tensor_values_with_unmapped_parallel_grid(tmp_path, mode):
    result = capture(tmp_path, mode)
    assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    proofs = [row for row in result.trace["transformations"] if row["kind"] == "source-tile-write-partition"]
    assert len(proofs) == 1 and proofs[0]["writes"] == (3 if mode == "nested" else 2)
    assert proofs[0]["observed_read_ids"]


@pytest.mark.parametrize("mode", ("hole", "independent", "different-index", "early", "repeat-program", "duplicate-lanes"))
def test_missing_lanes_early_observations_and_parallel_conflicts_stay_refused(tmp_path, mode):
    with pytest.raises(ConversionError):
        capture(tmp_path, mode)


@pytest.mark.parametrize("mode", MODES)
def test_native_tiles_keep_tails_full_control_words_and_all_observations(tmp_path, mode):
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
    root = tmp_path_factory.mktemp("tile-partition-" + request.param)
    result = capture(root, request.param)
    inputs, expected = arguments(request.param, 3)
    write_shared_bundle(result, root / "capture", inputs, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, request.param


def test_reusable_merlin_tile_library_preserves_every_result(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import verify_kernel_qualification

    root, mode = compiled
    library, receipt = _record(root / "merlin/host/model.so"), verify_kernel_qualification(root / "merlin")
    for index in range(5):
        inputs, expected = arguments(mode, index)
        arguments_file, reference_file = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
        np.savez(arguments_file, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference_file, **{f"out{i}": value for i, value in enumerate(expected)})
        out = root / f"invoke-{index}"
        report = invoke_kernel(root / "merlin", arguments_file, reference_file, out,
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(out)
    assert _record(root / "merlin/host/model.so") == library and verify_kernel_qualification(root / "merlin") == receipt


def test_actual_serial_images_keep_parallel_tile_algorithm_and_observations(compiled):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    root, mode = compiled
    for index in (0, 3):
        inputs, expected = arguments(mode, index)
        arguments_file, reference_file = root / f"target-arguments-{index}.npz", root / f"target-reference-{index}.npz"
        np.savez(arguments_file, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference_file, **{f"out{i}": value for i, value in enumerate(expected)})
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=arguments_file, reference=reference_file, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
