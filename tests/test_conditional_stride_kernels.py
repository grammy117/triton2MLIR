"""Selected memory steps retain tiles, original branches and every observation."""

import json
import os

import numpy as np
import pytest
from test_invariant_stride_kernels import TOOLS
from test_invariant_stride_kernels import arguments as invariant_arguments
from test_invariant_stride_kernels import capture as invariant_capture

from t2m.diagnostics import ConversionError

pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source/native tools required")
MODES = ("scalar", "pointer", "pure", "observe", "nested", "bit")


def capture(root, mode, transpose=False, dtype="i32", profile="tensor", policy="structured", **kwargs):
    return invariant_capture(root, mode, transpose, dtype, profile, policy, conditional_mode=mode, **kwargs)


def arguments(mode, transpose, dtype, index):
    return invariant_arguments(mode, transpose, dtype, index, conditional=True)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("transpose,dtype", [(False, "i32"), (True, "i64")])
@pytest.mark.parametrize("profile,policy", [("tensor", "structured"), ("buffer", "structured"),
    ("guarded-buffer", "structured"), ("guarded-buffer", "source-words")])
def test_actual_choices_and_local_division_preserve_tiles_and_final_reads(tmp_path, mode, transpose, dtype, profile, policy):
    from t2m.native import execute_native

    result = capture(tmp_path, mode, transpose, dtype, profile, policy)
    if policy == "structured":
        proof = next(row for row in result.trace["transformations"] if row["kind"] == "source-bounded-runtime-pointer-recurrences")
        assert [row["logical_unit_factor"] for row in proof["streams"]] == [1 if transpose else 17] * 2
        if mode not in ("scalar", "bit") or mode == "scalar" and not transpose:
            branch = next(row for row in result.trace["transformations"] if row["kind"] == "source-branch-logical-offsets")
            assert "no-speculation" in branch["evaluation"]
    if profile == "tensor":
        assert not any(op.name.startswith(("memref.", "gpu.")) for op in result.module.walk())
        assert "scf.forall" in result.mlir_text
        assert any(row.get("actual_bounded_recurrence_results") for row in result.trace["memory"]["source_loops"])
        if mode == "observe":
            assert any(row.get("actual_bounded_integer_results") for row in result.trace["memory"]["source_conditionals"])
    for index in (0, 1, 3, 6):
        arrays, expected = arguments(mode, transpose, dtype, index)
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native-{index}", python=os.environ["T2M_NATIVE_PYTHON"])
        for a, b in zip(actual, expected, strict=True):
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes())


@pytest.mark.parametrize("mode", MODES[:-1])
def test_bounded_choices_still_need_independent_access_and_final_root_bounds(tmp_path, mode):
    with pytest.raises(ConversionError):
        capture(tmp_path, mode, memory_bounds=(-16, 16))


@pytest.mark.parametrize("profile", ("tensor", "buffer"))
@pytest.mark.parametrize("through_loop", (False, True, "branch"))
def test_bounded_selected_coordinate_does_not_grant_parallel_write_ownership(tmp_path, profile, through_loop):
    from t2m.examples.conditional_stride_kernels import (
        conflicting_branch_loop_choice,
        conflicting_choice,
        conflicting_loop_choice,
    )
    from t2m.frontend import capture_ttir
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

    source = tmp_path / "source.ttir"
    kernel = conflicting_branch_loop_choice if through_loop == "branch" else conflicting_loop_choice if through_loop else conflicting_choice
    source.write_text(capture_ttir(kernel, {"x": "*i64", "out": "*i64", "B": "constexpr", **({"upper": "i32"} if through_loop else {})}, {"B": 8}))
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", (8,), "read", "i64"),
        TensorBuffer("out", (32,) if through_loop else (16,), "readwrite" if through_loop else "write", "i64")), grid=(3,),
        runtime_scalars=(RuntimeScalar("upper", "i32", 2, bounds=(1, 1)),) if through_loop else ())
    with pytest.raises(ConversionError, match="parallel|ownership"):
        convert_shared(source, abi, tmp_path / "conversion", optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile=profile)


@pytest.fixture(scope="module", params=[(m, t, d) for m in MODES for t, d in ((False, "i32"), (True, "i64"))])
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(k) for k in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing Merlin tools required")
    mode, transpose, dtype = request.param
    root = tmp_path_factory.mktemp(f"conditional-stride-{mode}-{transpose}-{dtype}")
    result = capture(root, mode, transpose, dtype)
    arrays, expected = arguments(mode, transpose, dtype, 1)
    write_shared_bundle(result, root / "capture", arrays, expected, work=root / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    differences = json.loads((root / "capture/compatibility.json").read_text())["differences"]
    assert any("bounded per-iteration pointer increments" in row for row in differences)
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    return root, mode, transpose, dtype


def test_unchanged_library_replays_selected_progress_and_all_memory_observations(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    root, mode, transpose, dtype = compiled
    library = _record(root / "merlin/host/model.so")
    for index in range(7):
        arrays, expected = arguments(mode, transpose, dtype, index)
        inputs, golden = root / f"arguments-{index}.npz", root / f"reference-{index}.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(golden, **{f"out{i}": value for i, value in enumerate(expected)})
        report = invoke_kernel(root / "merlin", inputs, golden, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["functional_inputs_preserved"]
        verify_invocation(root / f"invoke-{index}")
    assert _record(root / "merlin/host/model.so") == library


def test_actual_serial_images_retain_every_selected_increment_and_zero_trip(compiled):
    from test_spike_model import TOOLS as TARGET_TOOLS
    from test_spike_model import Q

    if not TARGET_TOOLS:
        pytest.skip("existing serial tools required")
    root, mode, transpose, dtype = compiled
    for index in (0, 3):
        arrays, expected = arguments(mode, transpose, dtype, index)
        inputs, golden = root / f"target-arguments-{index}.npz", root / f"target-reference-{index}.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(golden, **{f"out{i}": value for i, value in enumerate(expected)})
        out = root / f"target-{index}"
        report = Q.qualify(root / "capture", out, inputs=inputs, reference=golden, comparison="words",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        assert report["evaluation"]["result_count"] == 2
        assert Q.verify(out, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"]) == report
