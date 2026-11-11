"""Packed source reads retain actual groups, mutation snapshots and root tags."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT",
                                            "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing explicit native/Merlin tools required")


def capture(root, dtype, warps, while_, selected):
    from t2m.examples.grouped_memory_kernels import grouped_read_state
    from t2m.frontend import capture_ttir
    names = ("x", "y", "out_initial", "out_sum", "out_final", "out_final_sum")
    pointer = "*fp32" if dtype == "f32" else "*" + dtype
    path = root / "source.ttir"
    path.write_text(capture_ttir(grouped_read_state, {**{name: pointer for name in names},
        "n": "i32", "capacity": "i32", "iterations": "i32", "delta": "i32",
        "B": "constexpr", "WHILE": "constexpr", "SELECT": "constexpr"}, {"B": 256, "WHILE": while_, "SELECT": selected}))
    spec = TensorABI("grouped_read_state", (
        TensorBuffer("x", ("capacity",), "readwrite", dtype), TensorBuffer("y", ("capacity",), "read", dtype),
        *(TensorBuffer(name, ("n",), "write", dtype) for name in names[2:])), grid=(GridExtent("n", 256),),
        runtime_scalars=(RuntimeScalar("n", "i32", 6, bounds=(0, 531)), RuntimeScalar("capacity", "i32", 7, bounds=(0, 550)),
            RuntimeScalar("iterations", "i32", 8, bounds=(0, 3)), RuntimeScalar("delta", "i32", 9, bounds=(-3, 5))))
    return convert_shared(path, spec, root / "conversion", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer", address_policy="source-words",
        source_layout={"target": "cuda:80", "num_warps": warps, "threads_per_warp": 32, "num_ctas": 1})


def arguments(n, capacity, iterations, delta, dtype, warps, selected):
    dt = np.dtype({"i32": np.int32, "i64": np.int64, "f32": np.float32}[dtype])
    x = (np.arange(capacity) * 3 - 7).astype(dt)
    y = (np.arange(capacity) * -2 + 11).astype(dt)
    if dtype == "i64":
        x += np.int64(1 << 56)
        y -= np.int64(1 << 55)
    changed = x.copy()
    chosen = np.arange(capacity) % 3 == 0 if selected else np.zeros(capacity, dtype=bool)
    changed[:n] = np.where(chosen[:n], y[:n], x[:n]) + dt.type(delta)

    def read(state):
        initial, summed = np.empty(n, dtype=dt), np.empty(n, dtype=dt)
        def load(lane, step=0):
            if lane >= n or lane + step >= capacity or step >= 0 and lane // 256 != (lane + step) // 256:
                return dt.type(-7 if step == -1 else -3)
            root = y if chosen[lane] else state
            return root[lane + max(step, 0)]
        stride = warps * 32
        for start in range(0, n, 256):
            # Independent oracle for this declared initial source profile.
            # It never reads the frontend coordinate matrices or emitted IR.
            for group in range(256 // (2 * stride)):
                for lane in range(stride):
                    left, right = start + group * 2 * stride + lane, start + group * 2 * stride + lane + stride
                    a, b = load(left, -1), load(right, -1)
                    first, second = dt.type(a + b), dt.type(b - a)
                    for step in range(iterations):
                        first = dt.type(first + load(right, step))
                    if left < n: initial[left], summed[left] = a, first
                    if right < n: initial[right], summed[right] = b, second
        return initial, summed
    before, after = read(x), read(changed)
    arrays = [x, y, *(np.asarray(v, np.int32) for v in (n, capacity, iterations, delta))]
    return arrays, [changed, *before, *after]


@pytest.mark.parametrize("dtype,warps,while_,selected", [("i64", 4, False, False), ("f32", 4, True, False),
    ("i32", 2, False, True), ("i64", 4, True, True)])
def test_grouped_reads_preserve_snapshots_across_mutation_and_runtime_domains(tmp_path, dtype, warps, while_, selected):
    from test_integer_products import replay_library
    result = capture(tmp_path, dtype, warps, while_, selected)
    assert sum(t.get("memory") == "isolated-source-word-events-in-original-scalar-region"
               for t in result.trace["transformations"]) == 2
    assert len(result.trace["memory"]["source_gathers"]) == (12 if selected else 6)
    assert len(result.trace["memory"]["source_loops"]) == 2
    assert all(t["packing_proof"]["group_storage"] == "one-vector-per-result-slot"
               for t in result.trace["transformations"] if t.get("kind") == "ttir-elementwise-map")
    if dtype == "i64" and not selected:
        from t2m.diagnostics import ConversionError
        with pytest.raises(ConversionError, match="logical grouping"):
            convert_shared(tmp_path / "source.ttir", result.spec, tmp_path / "unselected-source",
                optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile="guarded-buffer", address_policy="source-words")
    cases = [arguments(n, capacity, trips, delta, dtype, warps, selected) for n, capacity, trips, delta in
        ((300, 307, 3, 2), (0, 0, 0, -3), (1, 1, 0, 5), (256, 256, 3, -3), (531, 550, 2, 1))]
    replay_library(tmp_path, result, cases)
    if dtype == "i64" and not selected:
        from t2m.consumer import invoke_merlin_host
        bad = tmp_path / "bad-inputs.npz"
        np.savez(bad, in0=np.empty(0, np.int64), in1=np.empty(0, np.int64), in2=np.asarray(1, np.int32),
            in3=np.asarray(0, np.int32), in4=np.asarray(0, np.int32), in5=np.asarray(1, np.int32))
        with pytest.raises(ValueError, match="outside logical allocation"):
            invoke_merlin_host(tmp_path / "merlin", bad, tmp_path / "bad-invocation",
                merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])


@pytest.mark.parametrize("dtype,warps", [("i64", 2), ("f32", 4)])
def test_packed_pointer_results_keep_each_slots_original_selector_and_mask(tmp_path, dtype, warps):
    from test_integer_products import replay_library
    from t2m.examples.grouped_memory_kernels import grouped_pointer_result
    from t2m.frontend import capture_ttir
    source = tmp_path / "source.ttir"
    pointer = "*fp32" if dtype == "f32" else "*i64"
    source.write_text(capture_ttir(grouped_pointer_result, {**{name: pointer for name in ("x", "y", "out")},
        "n": "i32", "B": "constexpr"}, {"B": 256}))
    spec = TensorABI("grouped_pointer_result", tuple(TensorBuffer(name, ("n",), "write" if name == "out" else "read", dtype)
        for name in ("x", "y", "out")), grid=(GridExtent("n", 256),),
        runtime_scalars=(RuntimeScalar("n", "i32", 3, bounds=(0, 531)),))
    result = convert_shared(source, spec, tmp_path / "conversion", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile="guarded-buffer", address_policy="source-words",
        source_layout={"target": "cuda:80", "num_warps": warps, "threads_per_warp": 32, "num_ctas": 1})
    cases = []
    for n in (300, 0, 1, 256, 531):
        arrays, _ = arguments(n, n, 0, 0, dtype, warps, False)
        x, y = arrays[:2]
        expected = np.full(n, -9, dtype=x.dtype)
        for lane in range(n):
            partner = (lane // 256) * 256 + ((lane % 256) ^ (warps * 32))
            if partner < n: expected[lane] = y[partner] if partner % 3 == 0 else x[partner]
        cases.append(([x, y, np.asarray(n, np.int32)], [expected]))
    replay_library(tmp_path, result, cases)
