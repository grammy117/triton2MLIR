"""Repeated-bin updates retain masks, fills, wide values and logical program tiles."""

import os

import numpy as np
import pytest

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
pytestmark = pytest.mark.skipif(not (OPT and NATIVE), reason="existing shared/native environments required")
PROFILES = [("tensor", True), ("buffer", True), ("guarded-buffer", True), ("guarded-buffer", False)]


def capture(root, profile, dtype, masked, row_view):
    from t2m.examples.histogram_kernels import histogram_tiles
    from t2m.frontend import capture_ttir
    source = root / "source.ttir"
    source.write_text(capture_ttir(histogram_tiles, {"x": "*" + dtype, "out": "*i32", "n": "i32", "tiles": "i32",
        "B": "constexpr", "NB": "constexpr", "MASKED": "constexpr", "ROW_VIEW": "constexpr"},
        {"B": 16, "NB": 8, "MASKED": masked, "ROW_VIEW": row_view}))
    abi = TensorABI("histogram_tiles", (TensorBuffer("x", ("n",), dtype=dtype),
        TensorBuffer("out", ("tiles", 8), "write", dtype="i32")), grid=(GridExtent("tiles", 1),),
        runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 31)), RuntimeScalar("tiles", "i32", 3, bounds=(0, 3))))
    return convert_shared(source, abi, root / "translation", optimizer=OPT, memory_profile=profile,
                          address_policy="source-words" if profile == "guarded-buffer" else "structured")


def arguments(n, tiles, dtype, masked, case):
    scalar = np.int64 if dtype == "i64" else np.int32
    limits = np.iinfo(scalar)
    pattern = np.asarray([0, 1, 1, 7, -1, 8, 2, limits.max, limits.min, 0, 3, 3, 7, -9, 1, 0], dtype=scalar)
    x = np.resize(np.roll(pattern, case), n)
    expected = np.zeros((tiles, 8), dtype=np.int32)
    for program in range(tiles):
        lane = program * 16 + np.arange(16)
        values = np.zeros(16, dtype=scalar)
        active = lane < n
        values[active] = x[lane[active]]
        selected = active & (lane % 3 != 1) if masked else np.ones(16, dtype=bool)
        selected &= (values >= 0) & (values < 8)
        expected[program] = np.bincount(values[selected].astype(np.int64), minlength=8).astype(np.int32)
    return [x, np.asarray(n, dtype=np.int32), np.asarray(tiles, dtype=np.int32)], expected


CASES = [(0, 0), (0, 2), (1, 1), (13, 1), (31, 3)]


@pytest.mark.parametrize("profile,row_view", PROFILES)
@pytest.mark.parametrize("dtype", ["i32", "i64"])
@pytest.mark.parametrize("masked", [False, True])
def test_histogram_keeps_repeated_counts_masks_and_padding_values(tmp_path, profile, dtype, masked, row_view):
    from t2m.native import execute_native
    result = capture(tmp_path, profile, dtype, masked, row_view)
    assert any(t["kind"] == "ttir-histogram" and t["work"] == "elements-plus-bins" for t in result.trace["transformations"])
    assert any("prov.histogram_algorithm" in op.attributes for op in result.module.walk())
    for i, (n, tiles) in enumerate(CASES):
        arrays, expected = arguments(n, tiles, dtype, masked, i)
        (actual,), _ = execute_native(result.mlir_text, arrays, tmp_path / f"native{i}", python=NATIVE,
                                     shape_abi=result.spec.runtime_shapes())
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not MERLIN, reason="existing Merlin native toolchain required")
@pytest.mark.parametrize("profile,row_view", PROFILES)
@pytest.mark.parametrize("dtype", ["i32", "i64"])
@pytest.mark.parametrize("masked", [False, True])
def test_one_merlin_library_retains_histogram_counts_and_tile_domain(tmp_path, profile, dtype, masked, row_view):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle
    result = capture(tmp_path, profile, dtype, masked, row_view)
    arrays, expected = arguments(13, 1, dtype, masked, 0)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, [expected], work=tmp_path / "native", python=NATIVE)
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt, library = verify_kernel_qualification(compiled), _record(compiled / "host/model.so")
    for i, (n, tiles) in enumerate(CASES):
        arrays, expected = arguments(n, tiles, dtype, masked, i)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, out0=expected)
        report = invoke_kernel(compiled, inputs, reference, tmp_path / f"invoke{i}", merlin_root=root, python=python)
        assert report["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt and _record(compiled / "host/model.so") == library


@pytest.mark.parametrize("profile", ["tensor", "buffer"])
def test_flat_pointer_to_rank_two_dynamic_output_retains_tensor_proof_limit(tmp_path, profile):
    from t2m.diagnostics import ConversionError
    with pytest.raises(ConversionError, match="dynamic dense descriptor rank differs"):
        capture(tmp_path, profile, "i64", True, False)


@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_histogram_demo_packages_integer_counts_and_original_source(tmp_path, profile):
    import json
    from t2m.bundle import verify_bundle
    from t2m.cli import main
    output = tmp_path / "demo"
    flags = ["--address-policy", "source-words"] if profile == "guarded-buffer" else []
    assert main(["shared-demo", "--kernel", "histogram", "--out", str(output), "--optimizer", OPT,
                 "--python", NATIVE, "--memory-profile", profile, *flags]) == 0
    verify_bundle(output)
    golden = np.load(output / "golden.npy", allow_pickle=False)
    assert golden.shape == (2, 8) and golden.dtype == np.int32
    trace = json.loads((output / "provenance.json").read_text())
    assert any(t["kind"] == "ttir-histogram" for t in trace["transformations"])
