"""Native pointer limitations retain checked tile computation and actual state."""

import hashlib
import json
import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

TOOLS = all(os.environ.get(name) for name in ("TRITON_SHARED_OPT", "T2M_NATIVE_PYTHON"))
pytestmark = pytest.mark.skipif(not TOOLS, reason="existing source and native tools required")
KINDS = ("different_root_choice", "different_root_loop", "while_progress")
CASES = ((0, 0, 8, 1, -2), (1, 1, 8, 3, 2), (17, 3, 1, 2, -2),
         (31, 0, 0, 1, 0), (17, 2, 7, 2, -1), (31, 3, 8, 1, 0), (1, 0, 1, 1, 1))


def capture(root, kind, profile="tensor", policy="structured", *, partial=False,
            static_width=None, control_dtype="i32"):
    from t2m.examples import root_state_kernels

    kernel = getattr(root_state_kernels, kind)
    constants = {"B": 8}
    if kind == "different_root_choice":
        constants["PARTIAL"] = partial
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, {
        **dict.fromkeys(("x", "y", "out", "last"), "*i64"),
        **dict.fromkeys(("lower", "upper", "stride", "memory_stride"), control_dtype),
        "N": "i32",
        **dict.fromkeys(constants, "constexpr"),
    }, constants))
    width = "n" if static_width is None else static_width
    abi = TensorABI(kind, (
        TensorBuffer("x", (64, width), "read", "i64"),
        TensorBuffer("y", (32, width), "read", "i64"),
        TensorBuffer("out", (width,), "write", "i64"),
        TensorBuffer("last", (width,), "write", "i64"),
    ), grid=(GridExtent("n", 8),) if static_width is None else ((static_width+7)//8,),
        scalar_constants=() if static_width is None else (static_width,), runtime_scalars=(
        *(RuntimeScalar(name, control_dtype, position, bounds=bounds)
          for position, (name, bounds) in enumerate(zip(
              ("lower", "upper", "stride", "memory_stride"),
              ((0, 3), (0, 8), (1, 3), (-2, 2)), strict=True), 4)),
        *((RuntimeScalar("n", "i32", 8, bounds=(0, 31)),) if static_width is None else ()),
    ))
    return convert_shared(source, abi, root / "translation",
        optimizer=os.environ["TRITON_SHARED_OPT"], memory_profile=profile, address_policy=policy)


def arguments(kind, case):
    n, lower, upper, stride, amount = case
    x = np.arange(64*n, dtype=np.int64).reshape(64, n)*37 + (1 << 58)
    y = -np.arange(32*n, dtype=np.int64).reshape(32, n)*19 - (1 << 57)
    x[1::2] *= -1
    expected = reference(kind, x, y, (lower, upper, stride, amount))
    arrays = [x, y, *(np.asarray(v, np.int32) for v in (lower, upper, stride, amount, n))]
    return arrays, expected


def reference(kind, x, y, controls):
    lower, upper, stride, amount = map(int, controls)
    state, position, selected = x[0].copy(), 16+lower, x
    for row in range(lower, upper, stride):
        if kind == "different_root_choice":
            state += x[16+row*amount] if row < 3 else y[row-2]
        else:
            state += selected[position]
            if kind == "while_progress":
                position += amount if row < 3 else 1
            elif row < 3:
                position += amount
            else:
                selected, position = y, row-2
    value = selected[position]
    return (state, state) if kind == "different_root_choice" else (state ^ value, value)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("profile,policy", (("tensor", "structured"), ("buffer", "structured"),
    ("guarded-buffer", "structured"), ("guarded-buffer", "source-words")))
def test_actual_roots_controls_dynamic_tails_and_all_results_execute(tmp_path, kind, profile, policy):
    from t2m.native import execute_native

    result = capture(tmp_path, kind, profile, policy)
    if profile == "tensor":
        assert not any(op.name.startswith(("memref.", "gpu.")) for op in result.module.walk())
        assert "scf.forall" in result.mlir_text
        assert ("scf.while" if kind == "while_progress" else "scf.for") in result.mlir_text
        assert any(row["mask"] == "source-mask-covers-complete-clipped-tile"
                   for row in result.trace["memory"]["source_scatters"])
    if policy == "structured":
        continuation = next(row for row in result.trace["transformations"]
                            if row["kind"] == "native-structured-memory-residual")
        stage = continuation["native_stage"]
        assert stage["stage_completed"] is False
        assert stage["verified_source"]["stage"] == "source"
        assert stage["verified_source"]["stage_completed"] is True
        for name, pin in stage["artifacts"].items():
            data = (tmp_path / "translation/tensor-ir" / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == pin["sha256"]
    for number in (0, 1, 3, 4):
        arrays, expected = arguments(kind, CASES[number])
        actual, _ = execute_native(result.mlir_text, arrays, tmp_path / f"native-{number}",
            python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=result.spec.runtime_shapes())
        assert len(actual) == len(expected)
        for got, wanted in zip(actual, expected, strict=True):
            assert (got.shape, got.dtype, got.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())


@pytest.mark.parametrize("profile", ("tensor", "buffer"))
def test_partial_predicates_cannot_claim_complete_fresh_output(tmp_path, profile):
    with pytest.raises(ConversionError, match="initialized|written|complete"):
        capture(tmp_path, "different_root_choice", profile, partial=True)
    assert not (tmp_path / "translation/model.mlir").exists()


@pytest.fixture(scope="module", params=KINDS)
def compiled(tmp_path_factory, request):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    if not all(os.environ.get(k) for k in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")):
        pytest.skip("existing Merlin compiler required")
    root, kind = tmp_path_factory.mktemp(request.param), request.param
    result = capture(root, kind)
    arrays, expected = arguments(kind, CASES[4])
    write_shared_bundle(result, root / "capture", arrays, expected, work=root / "native",
        python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"])
    return root, kind


def test_one_merlin_library_retains_all_runtime_shapes_and_root_states(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    root, kind = compiled
    library = _record(root / "merlin/host/model.so")
    for number, case in enumerate(CASES):
        arrays, expected = arguments(kind, case)
        inputs, reference = root / f"inputs-{number}.npz", root / f"reference-{number}.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        invoke_kernel(root / "merlin", inputs, reference, root / f"invoke-{number}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(root / f"invoke-{number}")
    assert _record(root / "merlin/host/model.so") == library


def test_actual_serial_images_retain_empty_and_ragged_domains(compiled):
    from test_spike_model import TOOLS as SERIAL_TOOLS
    from test_spike_model import Q

    if not SERIAL_TOOLS:
        pytest.skip("existing cross compiler and Spike required")
    root, kind = compiled
    objects = []
    for number in (0, 4):
        inputs, reference = root / f"inputs-{number}.npz", root / f"reference-{number}.npz"
        # This test also runs independently of the host replay test.
        arrays, expected = arguments(kind, CASES[number])
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        target = root / f"target-{number}"
        report = Q.qualify(root / "capture", target, merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"], inputs=inputs, reference=reference, comparison="words")
        assert report["evaluation"]["result_count"] == 2
        build = json.loads((target / "build.json").read_text())["build"]
        assert build["runtime_invocation_abi"]["mlir_argument_specialization"] == "not-performed"
        objects.append(build["host_artifacts"]["object"]["sha256"])
        assert Q.verify(target, merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"]) == report
    assert len(set(objects)) == 1
