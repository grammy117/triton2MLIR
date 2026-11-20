"""Actual branch successors preserve prefix observations and conditional math."""

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
MODES = ("branch", "prefix-observed", "branch-observed", "nested", "guarded", "redundant", "one-arm", "two-roots")


def capture(root, mode):
    from t2m.examples.branch_tile_kernels import branch_partition

    source = root / "source.ttir"
    source.write_text(capture_ttir(branch_partition,
        {**{name: "*i64" for name in ("x", "bias", "out", "seen")}, "n": "i32", "control": "i32",
         "B": "constexpr", "MODE": "constexpr"}, {"B": 8, "MODE": mode}))
    abi = TensorABI("branch_partition", (TensorBuffer("x", ("n",), "read", "i64"),
        TensorBuffer("bias", (1,), "read", "i64"), TensorBuffer("out", ("n",), "write", "i64"),
        TensorBuffer("seen", ("n",), "write", "i64")), grid=(GridExtent("n", 8),),
        runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 31)),
                         RuntimeScalar("control", "i32", 5, bounds=(-(1 << 31), (1 << 31)-1))))
    return convert_shared(source, abi, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])


def reference(x, bias, control, mode):
    word = x ^ bias[0]
    p, q = word > 0, control > 0 if mode == "guarded" else control != 0
    first = word ^ 1
    if mode == "one-arm":
        result = word ^ (2 if q else 1)
    elif mode == "guarded":
        negative = (word & 65535) // control if q else word ^ 4
        result = np.where(p, first, negative)
    else:
        negative = word ^ (8 if mode == "nested" and control < 0 else 2 if q else 4)
        result = np.where(p, first, negative)
    if mode == "two-roots":
        seen = np.where(p, (first ^ word ^ 16), word ^ (32 if q else 64))
    elif mode in ("prefix-observed", "branch-observed"):
        seen = np.where(p, first, -31 if mode == "branch-observed" and not q else -17)
    else:
        seen = result.copy()
    return [result, seen]


def arguments(mode, index):
    n = (0, 1, 8, 17, 31, 9)[index]
    control = (0, -(1 << 31), -1, (1 << 31)-1, 0, 3)[index]
    x = np.arange(n, dtype=np.int64) * 37 + (1 << 56)
    x[1::2] *= -1
    if n:
        x[-1] = (0, (1 << 63)-1, -(1 << 63), -1, 0, -65535)[index]
    bias = np.asarray([(1 << 55) + index], np.int64)
    return [x, bias, np.asarray(n, np.int32), np.asarray(control, np.int32)], reference(x, bias, control, mode)


@pytest.mark.parametrize("mode", MODES)
def test_selected_branch_paths_retain_tensor_dataflow_and_source_correspondence(tmp_path, mode):
    result = capture(tmp_path, mode)
    assert "scf.forall" in result.mlir_text and "gpu." not in result.mlir_text
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    proofs = [row for row in result.trace["transformations"] if row["kind"] == "source-tile-prefix-distribution"]
    assert proofs and all(row["execution"].startswith("exactly-one-original-scalar-if-successor") for row in proofs)
    if mode == "two-roots":
        assert {row["root"] for row in proofs} == {"out", "seen"}
    if mode == "prefix-observed":
        assert any(row["observed_read_ids"] for row in proofs)
    if mode == "guarded":
        divisions = [op for op in result.module.walk() if op.name == "arith.divsi"]
        assert divisions
        for division in divisions:
            ancestors, parent = [], division.parent_op()
            while parent is not None:
                ancestors.append(parent.name)
                parent = parent.parent_op()
            assert "scf.if" in ancestors


@pytest.mark.parametrize("mode", ("missing-arm", "early-prefix", "early-branch", "wrong-index", "different-root", "repeat-program"))
def test_missing_paths_early_reads_addresses_and_program_conflicts_still_refuse(tmp_path, mode):
    with pytest.raises(ConversionError):
        capture(tmp_path, mode)


@pytest.mark.parametrize("mode", MODES)
def test_native_branch_choices_keep_masks_fills_and_inactive_division(tmp_path, mode):
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
    root = tmp_path_factory.mktemp("branch-" + request.param)
    result = capture(root, request.param)
    inputs, expected = arguments(request.param, 3)
    write_shared_bundle(result, root / "capture", inputs, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, request.param


def test_one_merlin_library_preserves_all_actual_branch_choices_and_reused_words(compiled):
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


def test_serial_images_keep_prefix_observations_and_selected_region_arithmetic(compiled):
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
