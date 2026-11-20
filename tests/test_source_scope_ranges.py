"""Actual loop reachability refines local index facts without leaking outward."""

import os
from dataclasses import replace

import numpy as np
import pytest
from test_mutating_view_models import implementation

from t2m.diagnostics import ConversionError
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_indexing import check_source_addresses
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared, parse_shared
from t2m.upstream import lower_with_shared

pytestmark = pytest.mark.skipif(not os.environ.get("TRITON_SHARED_OPT"), reason="existing shared optimizer required")


@pytest.mark.skipif(not os.environ.get("TRITON_SHARED_OPT"), reason="existing shared optimizer required")
def test_rank_three_matrix_views_keep_the_tensor_representation(tmp_path):
    result = implementation(tmp_path / "kernel", "permuted-base", profile="tensor").implementation
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    assert any(op.name == "tensor.insert" for op in result.module.walk())
    assert any(row["kind"] == "source-loop-body-ranges" for row in result.trace["transformations"])


def source(root, *, outside=False, after=False, upper="product", step=1, remainder=False):
    import triton
    import triton.language as tl

    from t2m.frontend import capture_ttir

    @triton.jit
    def scoped(x, out, rows, cols, OUTSIDE: tl.constexpr, AFTER: tl.constexpr, UPPER: tl.constexpr,
               STEP: tl.constexpr, REMAINDER: tl.constexpr):
        if OUTSIDE:
            displacement = 1 % cols if REMAINDER else 1 // cols
        else:
            displacement = 0
        count = rows * cols if UPPER == "product" else rows + cols
        for i in range(0, count, STEP):
            row, col = i // cols, i % cols
            value = tl.load(x + row * cols + col + displacement)
            tl.store(out + i, value ^ (i.to(tl.int64) * 3 + 1))
        if AFTER:
            displacement = 1 % cols if REMAINDER else 1 // cols
            value = tl.load(x + displacement)
            tl.store(out, value)

    text = capture_ttir(scoped, {"x":"*i64", "out":"*i64", "rows":"i32", "cols":"i32",
        "OUTSIDE":"constexpr", "AFTER":"constexpr", "UPPER":"constexpr", "STEP":"constexpr", "REMAINDER":"constexpr"},
        {"OUTSIDE":outside, "AFTER":after, "UPPER":upper, "STEP":step, "REMAINDER":remainder})
    path = root / "source.ttir"
    path.write_text(text)
    lower_with_shared(path, root / "source", optimizer=os.environ["TRITON_SHARED_OPT"], stage="source")
    abi = TensorABI("scoped", (TensorBuffer("x", ("rows", "cols"), "read", "i64"),
        TensorBuffer("out", ("rows", "cols"), "write", "i64")), runtime_scalars=(
            RuntimeScalar("rows", "i32", 2, bounds=(0, 9)), RuntimeScalar("cols", "i32", 3, bounds=(0, 11))))
    return (root / "source/lowered.mlir").read_text(), abi


def test_positive_product_reachability_keeps_zero_in_the_entry_domain(tmp_path):
    text, abi = source(tmp_path)
    proofs = check_source_addresses(parse_shared(text), abi)
    assert proofs
    assert all(scalar.bounds[0] == 0 for scalar in abi.runtime_scalars)


@pytest.mark.skipif(not os.environ.get("T2M_NATIVE_PYTHON"), reason="existing native toolchain required")
def test_actual_source_division_and_remainder_keep_empty_and_nonempty_domains(tmp_path):
    from t2m.native import execute_native

    _text, abi = source(tmp_path)
    result = convert_shared(tmp_path / "source.ttir", abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])
    assert not any(op.name.startswith("memref.") for op in result.module.walk())
    for index, (rows, cols) in enumerate(((0, 0), (0, 7), (3, 0), (1, 1), (4, 7), (9, 11))):
        initial = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols) + (1 << 55)
        expected = initial ^ (np.arange(rows * cols, dtype=np.int64).reshape(rows, cols) * 3 + 1)
        actual, _ = execute_native(result.mlir_text, [initial, np.asarray(rows, np.int32), np.asarray(cols, np.int32)],
            tmp_path / f"native-{index}", python=os.environ["T2M_NATIVE_PYTHON"], shape_abi=abi.runtime_shapes())
        assert len(actual) == 1
        assert (actual[0].shape, actual[0].dtype, actual[0].tobytes()) == (expected.shape, expected.dtype, expected.tobytes())


@pytest.mark.skipif(not all(os.environ.get(name) for name in ("T2M_NATIVE_PYTHON", "T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")),
                    reason="existing native/Merlin tools required")
def test_one_merlin_library_preserves_region_facts_and_empty_source_domains(tmp_path):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle

    _text, abi = source(tmp_path)
    result = convert_shared(tmp_path / "source.ttir", abi, tmp_path / "translation", optimizer=os.environ["TRITON_SHARED_OPT"])

    def arrays(rows, cols):
        initial = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols) + (1 << 55)
        expected = initial ^ (np.arange(rows * cols, dtype=np.int64).reshape(rows, cols) * 3 + 1)
        return [initial, np.asarray(rows, np.int32), np.asarray(cols, np.int32)], expected

    values, expected = arrays(4, 7)
    capture, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, capture, values, [expected], work=tmp_path / "native", python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(capture, compiled, merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"])
    library, receipt = _record(compiled / "host/model.so"), verify_kernel_qualification(compiled)
    for index, shape in enumerate(((0, 0), (0, 7), (3, 0), (1, 1), (4, 7), (9, 11))):
        values, expected = arrays(*shape)
        arguments, reference = tmp_path / f"arguments-{index}.npz", tmp_path / f"reference-{index}.npz"
        np.savez(arguments, **{f"in{i}": value for i, value in enumerate(values)})
        np.savez(reference, out0=expected)
        report = invoke_kernel(compiled, arguments, reference, tmp_path / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(tmp_path / f"invoke-{index}")
        assert report["evaluation"]["functional_inputs_preserved"]
        with np.load(tmp_path / f"invoke-{index}/evaluation/outputs.npz", allow_pickle=False) as actual:
            assert actual.files == ["out0"]
            assert (actual["out0"].shape, actual["out0"].dtype, actual["out0"].tobytes()) == (
                expected.shape, expected.dtype, expected.tobytes())
    assert _record(compiled / "host/model.so") == library and verify_kernel_qualification(compiled) == receipt


def test_body_reachability_does_not_hide_overflow_in_the_earlier_bound_product(tmp_path):
    text, abi = source(tmp_path)
    abi = replace(abi, runtime_scalars=(replace(abi.runtime_scalars[0], bounds=(0, 1 << 30)), abi.runtime_scalars[1]))
    with pytest.raises(ConversionError, match="overflow"):
        check_source_addresses(parse_shared(text), abi)


@pytest.mark.parametrize("after", (False, True))
def test_nested_product_body_facts_do_not_escape_to_the_outer_body(tmp_path, after):
    import triton
    import triton.language as tl

    from t2m.frontend import capture_ttir

    @triton.jit
    def nested(x, out, rows, cols, AFTER: tl.constexpr):
        for i in range(rows):
            for j in range(i * cols):
                row, col = j // cols, j % cols
                value = tl.load(x + row * cols + col)
                tl.store(out + j, value)
            if AFTER:
                value = tl.load(x + 1 // cols)
                tl.store(out, value)

    path = tmp_path / "source.ttir"
    path.write_text(capture_ttir(nested, {"x":"*i64", "out":"*i64", "rows":"i32", "cols":"i32", "AFTER":"constexpr"}, {"AFTER": after}))
    lower_with_shared(path, tmp_path / "source", optimizer=os.environ["TRITON_SHARED_OPT"], stage="source")
    module = parse_shared((tmp_path / "source/lowered.mlir").read_text())
    abi = TensorABI("nested", (TensorBuffer("x", ("rows", "cols"), "read", "i64"), TensorBuffer("out", ("rows", "cols"), "readwrite", "i64")),
        runtime_scalars=(RuntimeScalar("rows", "i32", 2, bounds=(0, 9)), RuntimeScalar("cols", "i32", 3, bounds=(0, 11))))
    if after:
        with pytest.raises(ConversionError, match="zero divisor"):
            check_source_addresses(module, abi)
    else:
        assert check_source_addresses(module, abi)


@pytest.mark.parametrize("options", ({"outside": True}, {"after": True}, {"upper": "sum"}))
@pytest.mark.parametrize("remainder", (False, True))
def test_body_facts_cannot_validate_earlier_later_or_unrelated_divisions(tmp_path, options, remainder):
    text, abi = source(tmp_path, **options, remainder=remainder)
    with pytest.raises(ConversionError, match="zero divisor"):
        check_source_addresses(parse_shared(text), abi)
