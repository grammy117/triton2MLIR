"""Exact address/stride cancellation never divides by a runtime extent."""

import pytest
from xdsl.dialects import arith, builtin
from xdsl.ir import Block

from t2m.diagnostics import ConversionError
from t2m.shared_indexing import Linear
from t2m.symbolic_layout import Polynomial as P, expression, recover_axes


def test_symbolic_zero_extent_cancels_without_runtime_division():
    n = P.affine(Linear(parameters=(("n", 0, 13, 1),)))
    row = Linear(4, (4, 0, 0))
    assert (P.affine(row) * n).divide_stride(n) == row
    assert P.constant(0).divide_stride(n) == Linear()
    with pytest.raises(ConversionError, match="exactly aligned"):
        (P.affine(row) * n + P.constant(1)).divide_stride(n)
    with pytest.raises(ConversionError, match="nonlinear"):
        (P.affine(row) * n * n).divide_stride(n)


def test_cancelled_address_still_checks_source_integer_width():
    block = Block(arg_types=[builtin.i32, builtin.i32])
    product = arith.MuliOp(*block.args)
    facts = {block.args[0]: Linear(parameters=(("n", 0, 65536, 1),)),
             block.args[1]: Linear(parameters=(("m", 0, 65536, 1),))}
    with pytest.raises(ConversionError, match="overflow"):
        expression(product.result, facts, (1,))
    product = arith.MuliOp(arith.ExtSIOp(block.args[0], builtin.i64),
                           arith.ExtSIOp(block.args[1], builtin.i64))
    assert expression(product.result, facts, (1,)).bounds((1,)) == (0, 2**32)


def test_symbolic_singleton_facts_fold_but_other_factors_do_not():
    assert P.affine(Linear(parameters=(("n", 1, 1, 1),))) == P.constant(1)
    with pytest.raises(ConversionError, match="positive monomial"):
        P.constant(0).divide_stride(P.constant(0))


def test_dynamic_stride_matching_retains_permuted_axes_and_refuses_wrong_layout():
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer
    buffer = TensorBuffer("out", ("m", "n"), "write")
    abi = TensorABI("layout", (buffer,), runtime_scalars=(
        RuntimeScalar("m", "i32", 0, bounds=(0, 9)), RuntimeScalar("n", "i32", 1, bounds=(0, 13))))
    block = Block(arg_types=[builtin.i32, builtin.i32])
    m, n = block.args
    facts = {m: abi.dimension("m"), n: abi.dimension("n")}
    forms, axes = recover_axes(abi, buffer, (8, 4), (1, n), (3, 0), facts)
    assert forms == (Linear(), Linear(3)) and axes == (1, 0)
    with pytest.raises(ConversionError, match="do not match"):
        recover_axes(abi, buffer, (4, 8), (m, 1), (0, 0), facts)
    with pytest.raises(ConversionError, match="rank differs"):
        recover_axes(abi, buffer, (4,), (1,), (0,), facts)


def test_uniform_singleton_origin_splits_exactly_without_runtime_division():
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer
    buffer = TensorBuffer("out", ("m", "n"), "write")
    abi = TensorABI("layout", (buffer,), runtime_scalars=(
        RuntimeScalar("m", "i32", 0, bounds=(0, 9)), RuntimeScalar("n", "i32", 1, bounds=(0, 13))))
    n = P.affine(abi.dimension("n"))
    row, column = Linear(programs=(1, 0, 0)), Linear(programs=(0, 8, 0))
    origin = P.affine(row) * n + P.affine(column)
    evidence = []
    forms, axes = recover_axes(abi, buffer, (1, 8), (0, 1), (origin, 0), {}, evidence=evidence)
    assert forms == (row, column) and axes == (0, 1)
    assert evidence[0]["kind"] == "exact-dense-uniform-origin-partition"
    # Reconstruct actual flat origins independently, including a zero stride.
    for width in (0, 1, 13):
        for r in range(9):
            for p1 in (0, 1):
                coordinates = [form.constant + form.programs[0] * r + form.programs[1] * p1 for form in forms]
                assert coordinates[0] * width + coordinates[1] == r * width + 8 * p1
    with pytest.raises(ConversionError, match="do not match"):
        recover_axes(abi, buffer, (4, 8), (0, 1), (origin, 0), {})


def test_uniform_origin_recovery_does_not_accept_nonlinear_coordinates():
    from t2m.scalar_abi import RuntimeScalar
    from t2m.shared_tensor import TensorABI, TensorBuffer
    buffer = TensorBuffer("out", ("m", "n"), "write")
    abi = TensorABI("layout", (buffer,), runtime_scalars=(
        RuntimeScalar("m", "i32", 0, bounds=(0, 9)), RuntimeScalar("n", "i32", 1, bounds=(0, 13))))
    n = P.affine(abi.dimension("n"))
    product = P.affine(Linear(programs=(1, 0, 0))) * P.affine(Linear(programs=(0, 1, 0)))
    with pytest.raises(ConversionError, match="nonlinear"):
        recover_axes(abi, buffer, (1, 8), (0, 1), (product * n, 0), {})
