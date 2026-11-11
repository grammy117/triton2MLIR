"""Mapped source memory keeps its logical pointer edge and actual event region."""

import os

import numpy as np
import pytest

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT",
                                            "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing explicit native/Merlin tools required")


def test_scalar_pointer_map_reaches_actual_merlin(tmp_path):
    from test_integer_products import replay_library

    from t2m.examples.elementwise_map_kernels import pointer_load_map
    from t2m.frontend import capture_ttir
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(pointer_load_map, {"x": "*fp32", "out": "*fp32"}, {}))
    abi = TensorABI("pointer_load_map", (TensorBuffer("x", (8,)), TensorBuffer("out", (8,), "write")))
    result = convert_shared(source, abi, tmp_path / "conversion", optimizer=os.environ["TRITON_SHARED_OPT"],
                            memory_profile="guarded-buffer", address_policy="source-words")
    assert any(t.get("memory") == "isolated-source-word-events-in-original-scalar-region"
               for t in result.trace["transformations"])
    cases = []
    for shift in (0, 3):
        x = np.arange(8, dtype=np.float32) - shift
        cases.append(([x], [x.copy()]))
    replay_library(tmp_path, result, cases)


def test_native_triton_rejects_map_stores_instead_of_granting_a_host_effect_profile(tmp_path):
    from t2m.diagnostics import ConversionError
    from t2m.frontend import inspect_ttir

    source = tmp_path / "source.ttir"
    source.write_text('''module {tt.func public @invalid(%x:!tt.ptr<i32>){
      %p=tt.splat %x:!tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
      %r="tt.map_elementwise"(%p) <{pack=1:i32}> ({
      ^bb0(%ptr:!tt.ptr<i32>):
        %v=tt.load %ptr:!tt.ptr<i32>
        tt.store %ptr,%v:!tt.ptr<i32>
        "tt.map_elementwise.return"(%v):(i32)->()
      }):(tensor<8x!tt.ptr<i32>>)->tensor<8xi32>
      tt.return
    }}''')
    with pytest.raises(ConversionError, match="native source verification failed"):
        inspect_ttir(source, tmp_path / "inspection")
    assert "Stores are not supported inside map_elementwise" in (tmp_path / "inspection/stderr.log").read_text()


def capture_state(tmp_path, dtype, while_):
    from t2m.examples.elementwise_map_kernels import memory_map_state
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = tmp_path / "source.ttir"
    pointer = "*fp32" if dtype == "f32" else "*" + dtype
    outputs = ("out_initial", "out_sum", "out_final", "out_final_sum")
    source.write_text(capture_ttir(memory_map_state, {
        "x": pointer, **{name: pointer for name in outputs}, "n": "i32", "capacity": "i32",
        "iterations": "i32", "delta": "i32", "B": "constexpr", "WHILE": "constexpr"}, {"B": 32, "WHILE": while_}))
    abi = TensorABI("memory_map_state", (TensorBuffer("x", ("capacity",), "readwrite", dtype),
        *(TensorBuffer(name, ("n",), "write", dtype) for name in outputs)),
        runtime_scalars=(RuntimeScalar("n", "i32", 5, bounds=(0, 31)),
        RuntimeScalar("capacity", "i32", 6, bounds=(0, 40)), RuntimeScalar("iterations", "i32", 7, bounds=(0, 4)),
        RuntimeScalar("delta", "i32", 8, bounds=(-3, 5))))
    return convert_shared(source, abi, tmp_path / "conversion", optimizer=os.environ["TRITON_SHARED_OPT"],
                            memory_profile="guarded-buffer", address_policy="source-words")


@pytest.mark.parametrize("dtype", ("i32", "i64", "f32"))
@pytest.mark.parametrize("while_", (False, True))
def test_mapped_branches_and_memory_loops_retain_initial_snapshots_and_whole_state(tmp_path, dtype, while_):
    from test_integer_products import replay_library

    result = capture_state(tmp_path, dtype, while_)
    numpy_type = {"i32": np.int32, "i64": np.int64, "f32": np.float32}[dtype]
    cases = []
    for n, capacity, iterations, delta in ((13, 20, 3, 2), (0, 0, 0, -1), (0, 9, 4, 5), (1, 1, 0, 5), (31, 31, 4, -3)):
        x = (np.arange(capacity) * 3 - 7).astype(numpy_type)
        if dtype == "i64":
            x += np.int64(1 << 60)
        changed = x.copy()
        changed[:n] += numpy_type(delta)

        def read(state, n=n, capacity=capacity, iterations=iterations):
            value = state[:n].copy()
            for step in range(iterations):
                for lane in range(n):
                    value[lane] += state[lane + step] if lane + step < capacity else numpy_type(-3)
            return value

        arrays = [x, *(np.asarray(v, np.int32) for v in (n, capacity, iterations, delta))]
        cases.append((arrays, [changed, x[:n].copy(), read(x), changed[:n].copy(), read(changed)]))
    replay_library(tmp_path, result, cases)
    if dtype == "i32" and not while_:
        from t2m.consumer import invoke_merlin_host

        bad = tmp_path / "bad-inputs.npz"
        np.savez(bad, in0=np.empty(0, np.int32), in1=np.asarray(1, np.int32), in2=np.asarray(0, np.int32),
                 in3=np.asarray(0, np.int32), in4=np.asarray(1, np.int32))
        with pytest.raises(ValueError, match="outside logical allocation"):
            invoke_merlin_host(tmp_path / "merlin", bad, tmp_path / "bad-invocation",
                              merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])


@pytest.mark.parametrize("dtype", ("i32", "i64", "f32"))
@pytest.mark.parametrize("stage", range(4))
def test_selected_map_arguments_results_and_scalar_reads_keep_their_original_root(tmp_path, dtype, stage):
    from test_integer_products import replay_library

    from t2m.examples.elementwise_map_kernels import selected_map_reads
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    pointer = "*fp32" if dtype == "f32" else "*" + dtype
    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(selected_map_reads, {**{name: pointer for name in ("x", "y", "out", "observed")},
        "n": "i32", "capacity_x": "i32", "capacity_y": "i32", "choice": "i1", "B": "constexpr", "STAGE": "constexpr"},
        {"B": 32, "STAGE": stage}))
    abi = TensorABI("selected_map_reads", (TensorBuffer("x", ("capacity_x",), dtype=dtype),
        TensorBuffer("y", ("capacity_y",), dtype=dtype), TensorBuffer("out", ("n",), "write", dtype),
        TensorBuffer("observed", ("n",), "write", dtype)), runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
        RuntimeScalar("capacity_x", "i32", 5, bounds=(0, 40)), RuntimeScalar("capacity_y", "i32", 6, bounds=(0, 40)),
        RuntimeScalar("choice", "i1", 7)))
    result = convert_shared(source, abi, tmp_path / "conversion", optimizer=os.environ["TRITON_SHARED_OPT"],
                            memory_profile="guarded-buffer", address_policy="source-words")
    assert any(t["kind"] == "selected-pointer-control" for t in result.trace["transformations"])
    cases = []
    numpy_type = {"i32": np.int32, "i64": np.int64, "f32": np.float32}[dtype]
    for n, capacity_x, capacity_y, choice in ((13, 20, 17, True), (0, 0, 0, False), (1, 1, 0, True),
                                             (1, 0, 1, False), (31, 31, 31, False)):
        x = (np.arange(capacity_x) * 17 + 3).astype(numpy_type)
        y = (np.arange(capacity_y) * -31 - 5).astype(numpy_type)
        if dtype == "i64":
            x += np.int64(1 << 60)
            y -= np.int64(1 << 59)
        expected = np.asarray([x[lane] if (lane % 2 == 0) == choice else y[lane] for lane in range(n)], dtype=numpy_type)
        arrays = [x, y, *(np.asarray(v, np.int32) for v in (n, capacity_x, capacity_y)), np.asarray(choice, np.bool_)]
        cases.append((arrays, [expected, expected.copy()]))
    replay_library(tmp_path, result, cases)
