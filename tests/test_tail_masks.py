import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType
from xdsl.ir import Block

from t2m.diagnostics import ConversionError
from t2m.shared_indexing import Linear
from t2m.tail_masks import TailExtent, recover_extent


def mask_case(limit=53, tile=16, *, wrong_start=False, runtime=False):
    block = Block(arg_types=[IndexType()] * (2 if runtime else 1))
    facts = {block.args[0]: Linear(0, (1, 0, 0))}
    if runtime:
        facts[block.args[1]] = Linear(parameters=(("n", 0, limit, 1),))

    def constant(n):
        op = arith.ConstantOp.from_int_and_width(n, IndexType())
        block.add_op(op)
        return op.result

    def binary(cls, a, b):
        op = cls(a, b)
        block.add_op(op)
        return op.result

    cap, end = constant(tile), block.args[1] if runtime else constant(limit)
    start = binary(arith.MuliOp, block.args[0], cap)
    next_ = binary(arith.AddiOp, start, cap)
    clipped = binary(arith.MinSIOp, next_, end)
    nonnegative = binary(arith.MaxSIOp, clipped, start)
    extent = binary(arith.SubiOp, nonnegative, constant(1) if wrong_start else start)
    extent = binary(arith.MinSIOp, cap, extent)
    return extent, facts, Linear(0, (tile, 0, 0))


@pytest.mark.parametrize("programs,limit", [(4, 53), (6, 53), (2**25, 53), (4, 64), (4, 9)])
def test_upstream_tail_normal_form_is_proved_without_grid_enumeration(programs, limit):
    value, facts, offset = mask_case(limit)
    extent = recover_extent(value, 16, [offset], facts, (programs,))
    if limit == 64:
        assert extent == 16  # Bounds prove the mask is full in every program.
        return
    assert isinstance(extent, TailExtent)
    assert extent.fits(offset, limit)
    assert not extent.fits(offset + Linear(1), limit)
    assert not extent.fits(offset, limit - 1)
    assert extent.as_dict()["limit"] == limit


def test_scalar_gated_read_retains_exact_extent_without_coverage_credit():
    from t2m.initialization import DefiniteWrites
    from t2m.shared_tensor import TensorABI, TensorBuffer
    from t2m.tail_masks import ReadSubsetExtent
    value, facts, offset = mask_case()
    block = value.owner.parent
    zero = arith.ConstantOp.from_int_and_width(0, IndexType())
    condition = arith.CmpiOp(block.args[0], zero, "eq")
    selected = arith.SelectOp(condition, value, zero)
    block.add_ops([zero, condition, selected])
    extent = recover_extent(selected.result, 16, [offset], facts, (4,), allow_read_subset=True)
    assert isinstance(extent, ReadSubsetExtent)
    assert extent.value is selected.result
    assert extent.fits_access(offset, 53, (4,))
    assert not extent.fits(offset, 53)
    assert extent.as_dict()["coverage"] == "read-bound-only-conditional-subset"
    with pytest.raises(ConversionError, match="unit-prefix"):
        recover_extent(selected.result, 16, [offset], facts, (4,))
    abi = TensorABI("gated", (TensorBuffer("out", (53,), "write"),), grid=(4,))
    initialization = DefiniteWrites(abi)
    initialization.store(abi.buffers[0], (offset,), (extent,))
    with pytest.raises(ConversionError, match="no proved complete overwrite"):
        initialization.finish()


def test_gated_constant_read_prefix_and_intermediate_overflow():
    from t2m.tail_masks import ReadSubsetExtent
    block = Block(arg_types=[IndexType()])
    zero = arith.ConstantOp.from_int_and_width(0, IndexType())
    cap = arith.ConstantOp.from_int_and_width(8, IndexType())
    condition = arith.CmpiOp(block.args[0], zero, "eq")
    selected = arith.SelectOp(condition, cap, zero)
    block.add_ops([zero, cap, condition, selected])
    extent = recover_extent(selected.result, 8, [Linear()], {}, (1,), allow_read_subset=True)
    assert isinstance(extent, ReadSubsetExtent) and extent.cap == 8
    assert extent.fits_access(Linear(), 8, (1,)) and not extent.fits(Linear(), 8)
    value, facts, offset = mask_case()
    selection = arith.SelectOp(condition, value, zero)
    with pytest.raises(ConversionError, match="overflow"):
        recover_extent(selection.result, 16, [offset], facts, (2**60,), allow_read_subset=True)


def test_nested_conditional_read_analysis_has_a_structural_budget():
    value, facts, offset = mask_case()
    block = value.owner.parent
    zero = arith.ConstantOp.from_int_and_width(0, IndexType())
    condition = arith.CmpiOp(block.args[0], zero, "eq")
    block.add_ops([zero, condition])
    for _ in range(33):
        selected = arith.SelectOp(condition, value, zero)
        block.add_op(selected)
        value = selected.result
    with pytest.raises(ConversionError, match="structural budget"):
        recover_extent(value, 16, [offset], facts, (4,), allow_read_subset=True)


def test_changed_subtraction_is_not_accepted_as_a_tail():
    value, facts, offset = mask_case(wrong_start=True)
    with pytest.raises(ConversionError, match="clipped affine prefix"):
        recover_extent(value, 16, [offset], facts, (4,))


def test_a_mask_for_another_address_is_not_a_bounds_proof():
    value, facts, offset = mask_case()
    # A shifted mask remains a real tail, but it must be tied to the actual
    # address and limit before an access is admitted by the tensor adapter.
    extent = recover_extent(value, 16, [offset + Linear(8)], facts, (4,))
    assert extent.limit == 61
    assert not extent.fits(offset + Linear(8), 53)


def test_tail_proof_checks_intermediate_index_overflow():
    value, facts, offset = mask_case()
    with pytest.raises(ConversionError, match="overflow"):
        recover_extent(value, 16, [offset], facts, (2**60,))


def test_guarded_runtime_limit_is_uniform_but_not_a_captured_constant():
    value, facts, offset = mask_case(limit=53, runtime=True)
    extent = recover_extent(value, 16, [offset], facts, (6,))
    assert isinstance(extent.limit, Linear)
    assert extent.limit.parameters == (("n", 0, 53, 1),)
    assert extent.fits(offset, 53)
    assert not extent.fits(offset, 52)
    assert not extent.fits(offset + Linear(1), 53)
    assert extent.as_dict()["limit"]["parameters"] == (("n", 0, 53, 1),)


def test_guarded_parameter_terms_cancel_only_when_their_identity_and_range_agree():
    n = Linear(parameters=(("n", 0, 53, 1),))
    assert n.dynamic and n.bounds((4,)) == (0, 53)
    assert n + n.scale(-1) == Linear()
    assert (n.scale(2) + Linear(4)).divide(2) == n + Linear(2)
    with pytest.raises(ConversionError, match="inconsistent guarded"):
        n + Linear(parameters=(("n", 0, 99, 1),))


def test_runtime_shift_cannot_be_mistaken_for_a_complete_fresh_output_partition():
    from t2m.initialization import DefiniteWrites
    from t2m.shared_tensor import TensorABI, TensorBuffer
    abi = TensorABI("shifted", (TensorBuffer("out", (16,), "write"),), grid=(2,))
    initialization = DefiniteWrites(abi)
    offset = Linear(0, (8, 0, 0), parameters=(("shift", 0, 1, 1),))
    initialization.store(abi.buffers[0], (offset,), (TailExtent(8, 16, offset, None),))
    with pytest.raises(ConversionError, match="no proved complete overwrite"):
        initialization.finish()


def test_row_local_mask_bounds_a_flattened_loop_address_without_partition_credit():
    value, facts, origin = mask_case(limit=13, tile=8)
    physical = origin + Linear(loops=((0, 0, 4, 13),))
    extent = recover_extent(value, 8, [physical], facts, (2,))
    assert extent.offset == origin
    assert extent.limit == 13
    assert extent.fits_access(physical, 65, (2,))
    assert not extent.fits_access(physical, 64, (2,))
    assert not extent.fits(physical, 65)  # Bounds alone do not prove coverage.


def test_an_access_bounds_proof_does_not_certify_complete_fresh_initialization():
    from t2m.initialization import DefiniteWrites
    from t2m.shared_tensor import TensorABI, TensorBuffer
    abi = TensorABI("holes", (TensorBuffer("out", (13,), "write"),), grid=(2,))
    offset = Linear(0, (8, 0, 0))
    shifted_origin = offset + Linear(1)
    extent = TailExtent(8, 13, shifted_origin, None)
    assert extent.fits_access(offset, 13, abi.grid)
    initialization = DefiniteWrites(abi)
    initialization.store(abi.buffers[0], (offset,), (extent,))
    with pytest.raises(ConversionError, match="no proved complete overwrite"):
        initialization.finish()
