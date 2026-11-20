"""Pure numeric regions remain executable inside partial tile memory spans."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/native tools required")
MODES = ("reduce", "zero-reduce", "tuple", "scan", "reverse", "map", "block")


def capture(root, mode):
    from t2m.examples.region_tile_kernels import region_partition

    source = root / "source.ttir"
    source.write_text(capture_ttir(region_partition,
        {**{name: "*i64" for name in ("x", "bias", "out", "seen")}, "n": "i32", "steps": "i32",
         "B": "constexpr", "MODE": "constexpr"}, {"B": 8, "MODE": mode}))
    abi = TensorABI("region_partition", (TensorBuffer("x", ("n",), "read", "i64"),
        TensorBuffer("bias", (1,), "read", "i64"), TensorBuffer("out", ("n",), "write", "i64"),
        TensorBuffer("seen", ("n",), "write", "i64")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
                         RuntimeScalar("steps", "i32", 5, bounds=(-2 if mode in (
                             "zero-reduce", "block") else 1, 4))))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def signed(word):
    word &= (1 << 64) - 1
    return word - (1 << 64) if word & (1 << 63) else word


def reference(x, bias, steps, mode):
    results, snapshots = [], []
    for start in range(0, len(x), 8):
        length = min(8, len(x) - start)
        valid = np.arange(8) < length
        word = np.zeros(8, np.int64)
        word[:length] = x[start:start + length]
        word ^= bias[0]
        positive = word > 0
        state, snapshot = word ^ 1, np.zeros_like(word)
        if mode == "block":
            earlier = np.where(valid & positive, state, -17)
            total = np.sum(np.where(valid, word, 0), dtype=np.int64)
            state = np.where(valid & ~positive, word ^ total ^ earlier, state)
            snapshot = np.where(valid, state, 0)
        else:
            for iteration in range(steps):
                earlier = np.where(valid & positive, state, -17)
                state = np.where(valid & ~positive, word ^ (iteration + 2) ^ earlier, state)
                observed = np.where(valid, state, 0)
                if mode == "tuple":
                    scale, offset = int(observed[0]), int(np.where(valid, word, 0)[0])
                    for a, b in zip(observed[1:], np.where(valid, word, 0)[1:], strict=True):
                        scale, offset = signed(scale * int(a)), signed(offset * int(a) + int(b))
                    mixed = np.int64(signed(scale ^ offset))
                elif mode in ("scan", "reverse"):
                    heads = valid & positive
                    scanned, flags = np.zeros(8, np.int64), np.zeros(8, np.int64)
                    previous, flag = 0, 0
                    for index in (range(7, -1, -1) if mode == "reverse" else range(8)):
                        previous = int(observed[index]) if heads[index] else signed(previous + int(observed[index]))
                        flag |= int(heads[index])
                        scanned[index], flags[index] = previous, flag
                    mixed = scanned ^ flags
                elif mode == "map":
                    divided = np.asarray([abs(int(value)) // 3 * (-1 if value < 0 else 1) for value in observed], np.int64)
                    mapped = divided + 3 * sum(range(iteration))
                    mixed = mapped ^ np.add(observed, np.int64(3))
                else:
                    mixed = np.sum(observed, dtype=np.int64)
                snapshot ^= observed ^ mixed
                state = np.where(valid & positive, observed ^ 8, state)
            if steps <= 0 and mode == "zero-reduce":
                state = np.where(valid & ~positive, word ^ 16, state)
        results.append(state[:length])
        snapshots.append(snapshot[:length])
    return [np.concatenate(results) if results else np.empty(0, np.int64),
            np.concatenate(snapshots) if snapshots else np.empty(0, np.int64)]


def arguments(mode, index):
    n = (0, 1, 8, 17, 31, 9)[index]
    steps = (0, -2, 1, 3, 0, 2)[index] if mode in ("zero-reduce", "block") else (1, 4, 1, 3, 4, 2)[index]
    x = np.arange(n, dtype=np.int64) * 37 + (1 << 56)
    x[1::2] *= -1
    if n:
        x[-1] = (0, (1 << 63)-1, -(1 << 63), -1, 0, -65535)[index]
    bias = np.asarray([(1 << 55) + index], np.int64)
    return [x, bias, np.asarray(n, np.int32), np.asarray(steps, np.int32)], reference(x, bias, steps, mode)


@pytest.mark.parametrize("mode", MODES)
def test_actual_pure_regions_retain_tile_dataflow_and_original_source_contracts(tmp_path, mode):
    result = capture(tmp_path, mode)
    assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    reports = [row for row in result.trace["transformations"] if row.get("retained_value_region_source_ids")]
    assert reports
    if mode == "block":
        assert any(row["kind"] == "source-tile-write-partition" for row in reports)
    else:
        assert any(row["kind"] == "source-loop-tile-state" for row in reports)
        assert any(op.name == "scf.for" and len(op.results) >= 2 for op in result.module.walk())
    if mode in ("reduce", "zero-reduce", "tuple", "block"):
        assert any(row["kind"] == "ttir-first-element-reduction" for row in result.trace["transformations"])
    elif mode in ("scan", "reverse"):
        assert any(row["kind"] == "associative-scan-network" for row in result.trace["transformations"])
    else:
        assert any(row["kind"] == "ttir-elementwise-map" for row in result.trace["transformations"])


@pytest.mark.parametrize("mode", MODES)
def test_native_region_composition_keeps_tuple_order_scan_direction_and_loop_state(tmp_path, mode):
    from t2m.native import execute_native

    result = capture(tmp_path, mode)
    for index in range(6):
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
    root = tmp_path_factory.mktemp("region-tile-" + request.param)
    result = capture(root, request.param)
    inputs, expected = arguments(request.param, 3)
    write_shared_bundle(result, root / "capture", inputs, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, request.param


def test_one_merlin_library_preserves_every_iteration_and_zero_trip_source_state(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import verify_kernel_qualification

    root, mode = compiled
    library, receipt = _record(root / "merlin/host/model.so"), verify_kernel_qualification(root / "merlin")
    for index in range(6):
        inputs, expected = arguments(mode, index)
        arguments_file, reference_file = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
        np.savez(arguments_file, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference_file, **{f"out{i}": value for i, value in enumerate(expected)})
        report = invoke_kernel(root / "merlin", arguments_file, reference_file, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(root / f"invoke-{index}")
    assert _record(root / "merlin/host/model.so") == library and verify_kernel_qualification(root / "merlin") == receipt


def test_actual_serial_images_preserve_loop_observations_and_source_zero_paths(compiled):
    from test_spike_model import TOOLS, Q

    if not TOOLS:
        pytest.skip("existing serial target tools required")
    root, mode = compiled
    for index in (4, 5):
        inputs, expected = arguments(mode, index)
        arguments_file, reference_file = root / f"target-arguments-{index}.npz", root / f"target-reference-{index}.npz"
        np.savez(arguments_file, **{f"in{i}": value for i, value in enumerate(inputs)})
        np.savez(reference_file, **{f"out{i}": value for i, value in enumerate(expected)})
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=arguments_file, reference=reference_file, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
