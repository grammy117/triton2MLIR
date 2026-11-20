"""Independent fresh roots preserve interleaving, observations and input updates."""

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
MODES = ("interleaved", "observed", "cross", "predicate", "readonly", "state")


def capture(root, mode):
    from t2m.examples.interleaved_tile_kernels import interleaved_partition

    source = root / "source.ttir"
    source.write_text(capture_ttir(interleaved_partition,
        {**{name: "*i64" for name in ("x", "bias", "a", "b", "seen_a", "seen_b")},
         "n": "i32", "control": "i32", "B": "constexpr", "MODE": "constexpr"}, {"B": 8, "MODE": mode}))
    abi = TensorABI("interleaved_partition", (TensorBuffer("x", ("n",), "readwrite" if mode == "state" else "read", "i64"),
        TensorBuffer("bias", (1,), "read", "i64"),
        *(TensorBuffer(name, ("n",), "write", "i64") for name in ("a", "b", "seen_a", "seen_b"))),
        grid=(GridExtent("n", 8),), runtime_scalars=(RuntimeScalar("n", "i32", 6, bounds=(0, 31)),
        RuntimeScalar("control", "i32", 7, bounds=(-(1 << 31), (1 << 31)-1))))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def reference(x, bias, control, mode):
    value = x ^ bias[0]
    p, q = x > 0, (x > 0) ^ (control == 0)
    first_a = value ^ 1
    early_a = np.where(p, first_a, -17)
    if mode == "predicate":
        q = early_a > 0
    first_b = early_a ^ value ^ 2 if mode in ("cross", "predicate") else value ^ 2
    early_b = np.where(q, first_b, -31)
    if mode == "state":
        state = value ^ 16
        value = state
    second_a = early_b ^ value ^ 4 if mode in ("cross", "predicate") else value ^ 4
    second_b = early_a ^ value ^ 8 if mode in ("cross", "predicate") else value ^ 8
    a, b = np.where(p, first_a, second_a), np.where(q, first_b, second_b)
    seen = (early_a, early_b) if mode in ("observed", "cross", "predicate") else (a.copy(), b.copy())
    return [*([state] if mode == "state" else []), a, b, *seen]


def arguments(mode, index):
    n = (0, 1, 8, 17, 31)[index]
    control = (0, -(1 << 31), -1, (1 << 31)-1, 0)[index]
    x = np.arange(n, dtype=np.int64) * 37 + (1 << 56)
    x[1::2] *= -1
    if n:
        x[-1] = (0, (1 << 63)-1, -(1 << 63), -1, 0)[index]
    bias = np.asarray([(1 << 55) + index], np.int64)
    return [x, bias, np.asarray(n, np.int32), np.asarray(control, np.int32)], reference(x, bias, control, mode)


@pytest.mark.parametrize("mode", MODES)
def test_independent_output_groups_keep_parallel_tensors_and_actual_root_evidence(tmp_path, mode):
    result = capture(tmp_path, mode)
    assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    proofs = [row for row in result.trace["transformations"] if row["kind"] == "source-tile-write-partition"]
    assert len(proofs) == 2
    assert all(row["independent_roots"] for row in proofs)
    if mode in ("observed", "cross", "predicate"):
        assert all(row["observed_read_ids"] for row in proofs)
    if mode == "readonly":
        assert all(row["unchanged_readonly_read_ids"] for row in proofs)
    if mode == "state":
        assert all(row["unchanged_other_access_ids"] for row in proofs)


@pytest.mark.parametrize("mode", ("hole", "early", "different-index", "alias-root", "repeat-program"))
def test_coverage_observation_alias_and_parallel_obligations_still_refuse(tmp_path, mode):
    with pytest.raises(ConversionError):
        capture(tmp_path, mode)


@pytest.mark.parametrize("mode", MODES)
def test_native_interleaving_keeps_each_result_and_all_source_words(tmp_path, mode):
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
    root = tmp_path_factory.mktemp("interleaved-" + request.param)
    result = capture(root, request.param)
    inputs, expected = arguments(request.param, 3)
    write_shared_bundle(result, root / "capture", inputs, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, request.param


def test_reusable_library_keeps_cross_output_dependencies_and_input_state(compiled):
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


def test_serial_images_keep_every_intermediate_observation_and_updated_state(compiled):
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
        assert report["evaluation"]["result_count"] == len(expected)
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
