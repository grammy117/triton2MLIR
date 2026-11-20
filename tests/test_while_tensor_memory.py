"""Both while regions carry original computation and complete tensor state."""

import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.examples.while_tensor_kernels import checked_while_update, tiled_while_update
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not os.environ.get("TRITON_SHARED_OPT"), reason="existing shared optimizer required")


def tiled(root, mode="preserve", dtype="i64"):
    root.mkdir(exist_ok=True)
    path = root / "source.ttir"
    word = "fp32" if dtype == "f32" else dtype
    path.write_text(capture_ttir(tiled_while_update,
        {**{name: "*" + word for name in ("x", "state", "before", "after")}, "n": "i32", "steps": "i32", "choose": "i1",
         "B": "constexpr", "MODE": "constexpr"}, {"B": 8, "MODE": mode}))
    abi = TensorABI("tiled_while_update", (TensorBuffer("x", ("n",), "read", dtype),
        TensorBuffer("state", ("n",), "write" if mode in ("preinit", "fresh") else "readwrite", dtype),
        TensorBuffer("before", ("n",), "write", dtype), TensorBuffer("after", ("n",), "write", dtype)),
        grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
            RuntimeScalar("steps", "i32", 5, bounds=(-2, 5)), RuntimeScalar("choose", "i1", 6)))
    return convert_shared(path, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def checked(root):
    root.mkdir(exist_ok=True)
    path = root / "source.ttir"
    path.write_text(capture_ttir(checked_while_update, {**{name: "*i64" for name in ("state", "checks", "before", "after", "condition_seen")},
        "limit": "i64", "step": "i64"}, {}))
    abi = TensorABI("checked_while_update", tuple(TensorBuffer(name, (1,),
        "readwrite" if name in ("state", "checks") else "write", "i64") for name in ("state", "checks", "before", "after", "condition_seen")),
        runtime_scalars=(RuntimeScalar("limit", "i64", 5, bounds=(-(1 << 60), 1 << 60)),
                         RuntimeScalar("step", "i64", 6, bounds=(1, 3))))
    return convert_shared(path, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


@pytest.mark.parametrize("mode", ("preserve", "preinit", "conditional"))
def test_real_while_memory_keeps_source_control_and_tensor_versions(tmp_path, mode):
    result = tiled(tmp_path, mode)
    assert any(op.name == "scf.while" for op in result.module.walk())
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    assert any(row.get("control") == "scf.while" for row in result.trace["memory"]["source_loops"])


def test_real_condition_memory_is_retained_in_both_while_regions(tmp_path):
    result = checked(tmp_path)
    loops = [op for op in result.module.walk() if op.name == "scf.while"]
    assert len(loops) == 1
    assert any(op.name == "tensor.insert_slice" for op in loops[0].before_region.walk())
    assert not any(op.name.startswith("memref.") for op in result.module.walk())


def test_body_only_initialization_cannot_supply_a_zero_trip_fresh_result(tmp_path):
    with pytest.raises(ConversionError, match="(initializ|complete overwrite)"):
        tiled(tmp_path, "fresh")


def test_unknown_source_loop_attributes_are_not_silently_dropped(tmp_path):
    source = tmp_path / "source.ttir"
    source.write_text('''module {
      tt.func public @annotated(%state:!tt.ptr<i64>,%n:i32) {
        %zero=arith.constant 0:i32
        %one=arith.constant 1:i32
        %three=arith.constant 3:i64
        %result=scf.while (%i=%zero):(i32)->i32 {
          %continue=arith.cmpi slt,%i,%n:i32
          scf.condition(%continue) %i:i32
        } do {
        ^bb0(%j:i32):
          %old=tt.load %state:!tt.ptr<i64>
          %next=arith.addi %old,%three:i64
          tt.store %state,%next:!tt.ptr<i64>
          %step=arith.addi %j,%one:i32
          scf.yield %step:i32
        } attributes {test.state_index=0:i64}
        tt.return
      }
    }''')
    abi = TensorABI("annotated", (TensorBuffer("state", (1,), "readwrite", "i64"),),
        runtime_scalars=(RuntimeScalar("n", "i32", 1, bounds=(0, 5)),))
    with pytest.raises(ConversionError, match="loop-attribute transport"):
        convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def tile_arguments(n, count, choose=True, *, mode="preserve", dtype="i64"):
    word = np.int64 if dtype == "i64" else np.float32
    x = np.arange(n, dtype=word) + (1 << 52) if dtype == "i64" else np.arange(n, dtype=word) / np.float32(3.14)
    old = np.arange(n, dtype=word) + (1 << 55) if dtype == "i64" else np.arange(n, dtype=word) * np.float32(7) + np.float32(99)
    current = np.zeros_like(old) if mode == "preinit" else old.copy()
    before = current.copy()
    for i in range(max(0, count)):
        if mode != "conditional" or choose:
            current = (current + x) + word(i)
    inputs = [x]
    if mode != "preinit":
        inputs.append(old)
    inputs.extend((np.asarray(n, np.int32), np.asarray(count, np.int32), np.asarray(choose)))
    return inputs, [current, before, current.copy()]


def checked_arguments(state, checks, limit, step):
    before = state
    trips = max(0, (limit - state + step - 1) // step)
    state += trips * step

    def word(value):
        return (value + (1 << 63)) % (1 << 64) - (1 << 63)

    result = [state, word(checks + trips + 1), before, state, word(checks + trips)]
    return [np.asarray([before], np.int64), np.asarray([checks], np.int64),
            np.asarray(limit, np.int64), np.asarray(step, np.int64)], [np.asarray([value], np.int64) for value in result]


def program_cases(program):
    if program == "checked":
        return [checked_arguments(*values) for values in ((7, 19, 7, 1), (1, 23, 6, 2), (9, -7, 5, 3),
            ((1 << 55), (1 << 56), (1 << 55) + 5, 1), ((1 << 63) - 1, (1 << 63) - 1, 0, 1))]
    return [tile_arguments(n, count, choose, mode=program) for n, count, choose in
            ((13, 3, True), (0, 5, True), (1, -2, True), (8, 0, True), (31, 5, False))]


def exact(actual, expected):
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected, strict=True):
        assert (observed.shape, observed.dtype, observed.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native tools required")
@pytest.mark.parametrize("dtype", ("i64", "f32"))
@pytest.mark.parametrize("mode", ("preserve", "preinit", "conditional"))
def test_native_tiles_keep_reuse_snapshots_empty_tails_and_zero_trips(tmp_path, mode, dtype):
    from t2m.native import execute_native

    result = tiled(tmp_path, mode, dtype)
    for i, (n, count, choose) in enumerate(((13, 3, True), (0, 5, True), (1, -2, True), (8, 0, True), (31, 5, False))):
        inputs, expected = tile_arguments(n, count, choose, mode=mode, dtype=dtype)
        actual, _ = execute_native(result.mlir_text, inputs, tmp_path / f"native-{i}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
        exact(actual, expected)


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native tools required")
def test_native_condition_observations_include_the_final_false_check(tmp_path):
    from t2m.native import execute_native

    result = checked(tmp_path)
    for i, values in enumerate(((7, 19, 7, 1), (1, 23, 6, 2), (9, -7, 5, 3),
        ((1 << 55), (1 << 56), (1 << 55) + 5, 1), ((1 << 63) - 1, (1 << 63) - 1, 0, 1))):
        inputs, expected = checked_arguments(*values)
        actual, _ = execute_native(result.mlir_text, inputs, tmp_path / f"native-{i}", python=os.environ["T2M_NATIVE_PYTHON"])
        exact(actual, expected)


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native tools required")
def test_distinct_before_after_numeric_interfaces_keep_original_forwarded_values(tmp_path):
    from t2m.native import execute_native

    source = tmp_path / "source.ttir"
    source.write_text('''module {
      tt.func public @different(%state:!tt.ptr<i64>,%out:!tt.ptr<i64>,%n:i32) {
        %zero=arith.constant 0:i32
        %one=arith.constant 1:i32
        %three=arith.constant 3:i64
        %result:2=scf.while (%i=%zero):(i32)->(i32,i64) {
          %old=tt.load %state:!tt.ptr<i64>
          %continue=arith.cmpi slt,%i,%n:i32
          scf.condition(%continue) %i,%old:i32,i64
        } do {
        ^bb0(%j:i32,%value:i64):
          %next=arith.addi %value,%three:i64
          tt.store %state,%next:!tt.ptr<i64>
          %step=arith.addi %j,%one:i32
          scf.yield %step:i32
        }
        tt.store %out,%result#1:!tt.ptr<i64>
        tt.return
      }
    }''')
    abi = TensorABI("different", (TensorBuffer("state", (1,), "readwrite", "i64"),
        TensorBuffer("out", (1,), "write", "i64")), runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(-2, 5)),))
    result = convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    assert any(row.get("control") == "scf.while" and row["before_state_arity"] != row["after_state_arity"]
               for row in result.trace["memory"]["source_loops"])
    for i, count in enumerate((-2, 0, 1, 5)):
        initial = np.asarray([(1 << 56) + 11], np.int64)
        expected = initial + 3 * max(0, count)
        actual, _ = execute_native(result.mlir_text, [initial, np.asarray(count, np.int32)],
            tmp_path / f"native-{i}", python=os.environ["T2M_NATIVE_PYTHON"])
        exact(actual, [expected, expected.copy()])


@pytest.mark.skipif(not all(os.environ.get(name) for name in ("T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")),
                    reason="existing native/Merlin tools required")
@pytest.mark.parametrize("program", ("preserve", "preinit", "conditional", "checked"))
def test_merlin_reuses_original_while_regions_and_complete_state(tmp_path, program):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle

    result = checked(tmp_path) if program == "checked" else tiled(tmp_path, program)
    cases = program_cases(program)
    write_shared_bundle(result, tmp_path / "capture", *cases[0], work=tmp_path / "native",
                        python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(tmp_path / "capture", tmp_path / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"])
    library, receipt = _record(tmp_path / "merlin/host/model.so"), verify_kernel_qualification(tmp_path / "merlin")
    for i, (inputs, expected) in enumerate(cases):
        arguments, reference = tmp_path / f"inputs-{i}.npz", tmp_path / f"reference-{i}.npz"
        np.savez(arguments, **{f"in{index}": value for index, value in enumerate(inputs)})
        np.savez(reference, **{f"out{index}": value for index, value in enumerate(expected)})
        report = invoke_kernel(tmp_path / "merlin", arguments, reference, tmp_path / f"invoke-{i}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(tmp_path / f"invoke-{i}")
    assert _record(tmp_path / "merlin/host/model.so") == library and verify_kernel_qualification(tmp_path / "merlin") == receipt


@pytest.mark.parametrize("program", ("preserve", "preinit", "conditional", "checked"))
def test_actual_serial_images_keep_zero_trips_updates_and_condition_effects(tmp_path, program):
    from test_spike_model import TOOLS, Q

    from t2m.shared_bundle import write_shared_bundle

    if not TOOLS or not os.environ.get("T2M_NATIVE_PYTHON"):
        pytest.skip("existing native/serial target tools required")
    result = checked(tmp_path) if program == "checked" else tiled(tmp_path, program)
    cases = program_cases(program)
    write_shared_bundle(result, tmp_path / "capture", *cases[0], work=tmp_path / "native",
                        python=os.environ["T2M_NATIVE_PYTHON"])
    for i in ((0, 1) if program == "checked" else (0, 3)):
        values, expected = cases[i]
        inputs, reference = tmp_path / f"inputs-{i}.npz", tmp_path / f"reference-{i}.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(values)})
        np.savez(reference, **{f"out{j}": value for j, value in enumerate(expected)})
        out = tmp_path / f"target-{i}"
        report = Q.qualify(tmp_path / "capture", out, inputs=inputs, reference=reference, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == len(expected)
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"],
                        python=os.environ["T2M_MERLIN_PYTHON"]) == report
