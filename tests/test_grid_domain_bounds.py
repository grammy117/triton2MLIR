"""Launch-derived memory bounds and fresh coverage remain separate obligations."""

import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
PROFILES = ["tensor", "buffer", "guarded-buffer"]
SCENARIOS = [pytest.param(profile, kind, marks=pytest.mark.xfail(strict=True, raises=ConversionError,
    reason="pinned unstructured converter leaves branch-local pointer preparation; tensor/buffer forms qualify"))
    if profile == "guarded-buffer" and kind == "conditional-update" else (profile, kind)
    for profile in PROFILES for kind in ("rows", "sparse-update", "conditional-update", "matrix")]


def capture(root, profile, *, matrix=False, step=1, shift=0, conditional=False, fresh=True,
            grid=None, flat=False, n_bounds=(0, 9)):
    from t2m.examples.grid_domain_kernels import row_cells, row_tail_matrix, flat_unmasked_tiles
    from t2m.frontend import capture_ttir
    if matrix:
        kernel, shape, constants = row_tail_matrix, ("m", "n"), {"B": 8, "SHIFT": shift}
        scalars = (RuntimeScalar("m", "i32", 2, bounds=(0, 5)), RuntimeScalar("n", "i32", 3, bounds=(0, 13)))
        launch = (GridExtent("m", 1), GridExtent("n", 8))
    else:
        kernel = flat_unmasked_tiles if flat else row_cells
        shape = ("n",) if flat else ("n", 8)
        constants = {"B": 8} if flat else {"B": 8, "STEP": step, "SHIFT": shift, "CONDITIONAL": conditional}
        scalars = (RuntimeScalar("n", "i32", 2, bounds=n_bounds), RuntimeScalar("launch", "i32", 3, bounds=(0, 9)))
        launch = (grid if grid is not None else GridExtent("n", 8 if flat else step),)
    source = root / "source.ttir"
    source.write_text(capture_ttir(kernel, {"x": "*fp32", "out": "*fp32", **{s.name: "i32" for s in scalars},
        **{name: "constexpr" for name in constants}}, constants))
    abi = TensorABI(kernel.__name__, (TensorBuffer("x", shape), TensorBuffer("out", shape, "write" if fresh else "readwrite")),
        grid=launch, runtime_scalars=scalars)
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile=profile,
                          address_policy="source-words" if profile == "guarded-buffer" else "structured")


def row_arguments(n, step=1, fresh=True, conditional=False):
    x = np.arange(n * 8, dtype=np.float32).reshape(n, 8) - 5
    incoming = np.full((n, 8), -111, dtype=np.float32)
    expected = incoming.copy()
    selected = np.arange(0, n, step)
    if conditional:
        selected = selected[np.arange(len(selected)) % 2 == 0]
    expected[selected] = x[selected] + 1
    return [x, *([] if fresh else [incoming]), np.asarray(n, dtype=np.int32), np.asarray(0, dtype=np.int32)], expected


def matrix_arguments(m, n):
    x = np.arange(m * n, dtype=np.float32).reshape(m, n) - 5
    return [x, np.asarray(m, dtype=np.int32), np.asarray(n, dtype=np.int32)], x + 1


def replay_library(tmp_path, result, cases):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    arrays, expected = cases[-1]
    write_shared_bundle(result, bundle, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, (arrays, expected) in enumerate(cases):
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        report = invoke_kernel(compiled, inputs, reference, tmp_path / f"invoke{i}", merlin_root=root, python=python)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library


@pytest.mark.parametrize("profile,kind", SCENARIOS)
def test_grid_bounds_retain_unmasked_rows_and_partial_state(tmp_path, profile, kind):
    from t2m.native import execute_native
    matrix = kind == "matrix"
    step, fresh, conditional = (2 if kind == "sparse-update" else 1), kind in ("rows", "matrix"), kind == "conditional-update"
    result = capture(tmp_path, profile, matrix=matrix, step=step, fresh=fresh, conditional=conditional)
    cases = ([matrix_arguments(m, n) for m, n in ((0, 0), (0, 7), (3, 0), (1, 1), (5, 13))] if matrix else
             [row_arguments(n, step, fresh, conditional) for n in (0, 1, 7, 9)])
    for i, (arrays, expected) in enumerate(cases):
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                     shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual, expected)
    if profile != "guarded-buffer" and kind == "rows":
        assert result.trace["memory"]["output_initialization"]["out"]["coverage"] == "disjoint-exact-runtime-grid-partition"


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("profile,kind", SCENARIOS)
def test_one_merlin_library_retains_grid_bounds_and_input_state(tmp_path, profile, kind):
    matrix = kind == "matrix"
    step, fresh, conditional = (2 if kind == "sparse-update" else 1), kind in ("rows", "matrix"), kind == "conditional-update"
    result = capture(tmp_path, profile, matrix=matrix, step=step, fresh=fresh, conditional=conditional)
    cases = ([matrix_arguments(m, n) for m, n in ((0, 0), (0, 7), (3, 0), (1, 1), (5, 13))] if matrix else
             [row_arguments(n, step, fresh, conditional) for n in (0, 1, 7, 9)])
    replay_library(tmp_path, result, cases)


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
@pytest.mark.parametrize("case", ["shift-forward", "shift-back", "other-dimension", "constant-launch", "holes", "branch-holes", "ragged"])
def test_invalid_access_and_missing_coverage_remain_refused(tmp_path, profile, case):
    options = {
        "shift-forward": {"shift": 1}, "shift-back": {"shift": -1},
        "other-dimension": {"grid": GridExtent("launch", 1)}, "constant-launch": {"grid": 3},
        "holes": {"step": 2}, "branch-holes": {"conditional": True}, "ragged": {"flat": True},
    }[case]
    with pytest.raises(ConversionError, match="bounds|complete overwrite"):
        capture(tmp_path, profile, **options)


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_exact_guarded_dimension_establishes_wider_tile_divisibility(tmp_path, profile):
    from t2m.native import execute_native
    result = capture(tmp_path, profile, flat=True, n_bounds=(16, 16))
    for i in range(2):
        x = np.arange(16, dtype=np.float32) + 100 * i
        (actual,), _ = execute_native(result.mlir_text, [x, np.asarray(16, dtype=np.int32), np.asarray(i, dtype=np.int32)],
            tmp_path / f"native{i}", python=NATIVE, shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual, x + 1)
