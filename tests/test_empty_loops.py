"""Wholly inactive source loops preserve entry values and active-read checks."""

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
MODES = ("after", "before", "complete", "address")


def capture(root, mode, profile="tensor", bounds=(-2, 0)):
    from t2m.examples.empty_loop_kernels import empty_recurrence

    source = root / "source.ttir"
    source.write_text(capture_ttir(empty_recurrence,
        {**{name: "*i64" for name in ("x", "bias", "out", "seen")}, "n": "i32", "steps": "i32",
         "B": "constexpr", "MODE": "constexpr"}, {"B": 8, "MODE": mode}))
    abi = TensorABI("empty_recurrence", (TensorBuffer("x", ("n",), "read", "i64"),
        TensorBuffer("bias", (1,), "read", "i64"), TensorBuffer("out", ("n",), "write", "i64"),
        TensorBuffer("seen", ("n",), "write", "i64")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
                         RuntimeScalar("steps", "i32", 5, bounds=bounds)))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile=profile)


def reference(x, bias, steps, mode):
    assert steps <= 0
    word = x ^ bias[0]
    result = word ^ 1 if mode == "complete" else np.where(word > 0, word ^ 1, word ^ 16)
    return [result, word ^ 64]


def arguments(mode, index):
    n = (0, 1, 8, 17, 31, 9)[index]
    steps = (0, -2, -1, 0, -2, -1)[index]
    x = np.arange(n, dtype=np.int64) * 37 + (1 << 56)
    x[1::2] *= -1
    if n:
        x[-1] = (0, (1 << 63)-1, -(1 << 63), -1, 0, -65535)[index]
    bias = np.asarray([(1 << 55) + index], np.int64)
    return [x, bias, np.asarray(n, np.int32), np.asarray(steps, np.int32)], reference(x, bias, steps, mode)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("profile", ("tensor", "buffer"))
def test_empty_source_bodies_grant_no_write_credit_and_preserve_original_initial_values(tmp_path, mode, profile):
    result = capture(tmp_path, mode, profile)
    assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    if profile == "tensor":
        assert not any(op.name.startswith("memref.") for op in result.module.walk())
    loops = result.trace["memory"]["source_loops"]
    assert any(row.get("maximum_iterations") == 0 and row.get("recurrence") == "complete-original-initial-tuple" for row in loops)
    if profile == "buffer":
        assert any(row["kind"] == "native-empty-loop-domain-guards" for row in result.trace["transformations"])
        assert "cf.assert" in result.auxiliary_artifacts["memory-input.mlir"]


@pytest.mark.parametrize("mode,bounds", [("early", (-2, 0)), ("unsafe-before", (-2, 0)), ("after", (-2, 1))])
def test_active_early_reads_live_definitions_and_positive_trips_do_not_borrow_empty_proofs(tmp_path, mode, bounds):
    with pytest.raises(ConversionError):
        capture(tmp_path, mode, bounds=bounds)


@pytest.mark.parametrize("profile", ("tensor", "buffer"))
def test_inactive_writes_do_not_initialize_a_fresh_result(tmp_path, profile):
    from t2m.examples.tensor_kernels import runtime_memory_store

    source = tmp_path / "source.ttir"
    source.write_text(capture_ttir(runtime_memory_store,
        {"x": "*fp32", "iterations": "i32", "out": "*fp32", "N": "constexpr", "B": "constexpr"},
        {"N": 13, "B": 8}))
    abi = TensorABI(runtime_memory_store.__name__, (TensorBuffer("x", (13,)), TensorBuffer("out", (13,), "write")),
        grid=(2,), runtime_scalars=(RuntimeScalar("iterations", "i32", 1, bounds=(-2, 0)),))
    with pytest.raises(ConversionError, match="no established store tile"):
        convert_shared(source, abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile=profile)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("profile", ("tensor", "buffer"))
def test_native_initial_results_and_nonpositive_upper_bounds(tmp_path, mode, profile):
    from t2m.native import execute_native

    result = capture(tmp_path, mode, profile)
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
    root = tmp_path_factory.mktemp("empty-loop-" + request.param)
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


def test_an_active_outside_access_cannot_disappear_with_its_inactive_body_user(tmp_path):
    from t2m.native import observe_native

    try:
        result = capture(tmp_path, "bounds")
    except ConversionError:
        return  # A source-located static refusal is also valid admission.
    inputs, _ = arguments("after", 3)
    inputs[-1] = np.asarray(0, np.int32)
    out = tmp_path / "outside-bounds-abort"
    report = observe_native(result.mlir_text, inputs, out,
        python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
    assert report["returncode"] == -6 and not (out / "outputs.npz").exists()
    assert (out / "invocation-started.json").exists()


def test_native_continuation_rejects_a_positive_trip_even_without_its_outer_scalar_guard(tmp_path):
    from xdsl.dialects import cf, func

    from t2m.ir import text
    from t2m.native import observe_native

    result = capture(tmp_path, "after", "buffer")
    module = result.module.clone()
    entry = next(op for op in module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == result.spec.entry)
    # This adversarial copy deliberately removes the public header assertions.
    # The actual independently guarded native continuation must still abort.
    guards = [op for op in entry.body.block.ops if isinstance(op, cf.AssertOp)]
    assert guards
    for guard in guards:
        entry.body.block.erase_op(guard)
    inputs, _ = arguments("after", 3)
    inputs[-1] = np.asarray(1, np.int32)
    shape_abi = result.spec.runtime_shapes()
    # Bypass only the test runner's scalar range preflight, so the invalid
    # value reaches the compiled continuation. Keep its actual shape contract.
    next(row for row in shape_abi["inputs"] if row.get("scalar") == "steps").pop("bounds")
    out = tmp_path / "private-domain-abort"
    report = observe_native(text(module), inputs, out,
        python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=shape_abi)
    assert report["returncode"] == -6 and not (out / "outputs.npz").exists()
    assert (out / "invocation-started.json").exists()
